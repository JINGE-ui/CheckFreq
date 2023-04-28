import torch
import os
import enum
import logging
import copy
from collections import OrderedDict
from collections.abc import Mapping
import time
"""
Checkpointing and restoring logic

"""


"""
Manages the model, optimizer and optionally dataloader state
For AUTO mode, dl state must be checkpointable - use CheckFreq iterator

`tracking_map` has the list of tractable objects provided by the user
to be checkpointed.
Each tractable item must expose a state_dict() method that returns 
a map of items to be checkpointed.
Common examples of such tractable objects are model and optimizer.

`num_tracking` excludes any additional state info that is
snapshotted outside of CF checkpointer. the DataLoader state 
is an example of such attribute.

CF checkpointer only tracks objects which will be snapshotted
as a part of the manager.

`latest_snapshot` if not None implies that there is an ongoing 
checkpointing operation. This holds the in-memory (GPU/CPU-memory)
copy of the state_dicts of the user-tractable objects.
If theese objects hold CUDA tensors, then the snapshot holds
a copy of the tensor on the same device as the original.
If the object is fully CPU resident, so is the snapshot.
This is set to None when the snapshot has been successfully 
persisted.

`latest_snapshot` is a map of copy (deepcopy) of state_dict of items in the `tracking_map` and any additional state managed by the
CFManager (such as DL state), global chk ID, etc

"""

class CFCheckpoint:

	def __init__(
		self,
		**kwargs):	
		self.logger = logging.getLogger(__name__)		#logger类： 提供应用程序直接使用的接口
		self.tracking_map = OrderedDict()		#待checkpoint的对象列表  有序字典
		self.latest_snapshot = None		#tracking_map列表中对象 的备份
		self.pipeline_snapshot = None

		for name, ref in kwargs.items():
			if hasattr(ref, 'state_dict'):
				self.tracking_map[name] = ref
			else:
				self.logger.info("Skipping object `{}` in CF Checkpointing. \
				No state_dict() method exposed".format(name))

		self.num_tracking = len(self.tracking_map.keys())	#待checkpoint的对象数目

		if self.num_tracking == 0:
			raise ValueError("Nothing to track")
		

	def __getstate__(self):
		d = self.__dict__.copy()   #__dict__属性：类调用输出类属性字典， 实例调用则是实例属性字典
		if 'logger' in d:
			d['logger'] = d['logger'].name
		return d

	def __setstate__(self, d):
		if 'logger' in d:
			d['logger'] = logging.getLogger(d['logger'])
		self.__dict__.update(d)


	"""
	Snapshots the current state of all tractable objects
	and any additional state, collectively as a map in `latest_snapshot`

	`additional state` is a map of items to be checkpointed
	Note that all objects in ``additional state` is assumed to be 
	a copy already - we do not snapshot it, rather only keep track
	of it to persist later.
	It is expected to be a map :
	For example
	additional_state = {
		'dl_state' : copy_of_dl_state,
		'chk_global_id' : 1,
		'donot_delete' : False			
	}

	This in-memory snapshotting is done synchronously

	Returns True on success, False otherwise

	"""
	
	def _snapshot(self, active_snapshot, additional_state=None):	#系统快照
		if active_snapshot == 1:
		#if self.latest_snapshot is not None:
			self.logger.info("Snapshot is not None. Checkpoint underway")
			return False
		
		self.latest_snapshot = OrderedDict()	# 若是对象是CUDA张量，则latest_snapshot也位于GPU上
	
		#Snapshot the state of tractable items
		for name, ref in self.tracking_map.items():
			if name not in self.latest_snapshot:
				self.latest_snapshot[name] = copy.deepcopy(ref.state_dict())
			else:
				self.logger.info("Repeated entry for {}".format(name))
				return False

		if isinstance(additional_state, Mapping):	#判断additional_state是否是map
			self.latest_snapshot.update(additional_state)	#将additional_state添加到latest_snapshot
				
		return True

	"""
	serializes the tensors (copies GPU tensors to CPU), and
	persists the snapshot to disk.

	Fsync the file after writing to gaurantee persistence

	Call this in a new process to perform in bgk
	"""	
	def _serialize_and_persist(
		self, 
		filepath,
		snapshot,
		active_snapshot,
		lock,
		linkpath=None,
		iter_chk = None,
		epoch_chk = None, 
		overwrite = True):
		print("[{}] START ASYNC".format(time.time()))

		with lock:
			if active_snapshot.value == 0:
				self.logger.error("Cannot persist. Empty snapshot")
				return
		#Create new stream
		s = torch.cuda.Stream()
		# 通过使用stream:
		# 1. 计算和数据拷贝可以同时进行
		# 2. 两个方向的拷贝可以同时进行（GPU到CPU，和CPU到GPU）
		torch.cuda.stream(s)	# 选择 s 流

		#print("Saving : {}".format(filepath))
		torch.save(snapshot, filepath)
		#print("Saved : {}".format(filepath))
		# Clear the snapshot.
		with lock:
			active_snapshot.value = 0

		# Ensure its persisted
		f = open(filepath, 'a+')
		os.fsync(f.fileno())
		f.close()

		update_stats(
				filepath,
				iter_chk=iter_chk,
				overwrite=overwrite,
				epoch_chk = epoch_chk,
				linkpath=linkpath)
		print("[{}] END ASYNC".format(time.time()))


	def _serialize_and_persist_direct(
		self,
		snapshot, 	# 所有备份对象的副本
		filepath, 	 # checkpoint文件保存地址
		additional_state=None, 	  # 额外需要备份的程序运行状态
		persist=True,	# true: 使用fsync强制刷新至硬盘
		linkpath=None,
		iter_chk = None,	# 所有iter-level的checkpoint文件名
		epoch_chk = None,	# 所有epoch边界的checkpoint文件名
		overwrite = True):

		snap_ptr = {}
		for name, ref in self.tracking_map.items():
				snap_ptr[name] = ref.state_dict()

		if isinstance(additional_state, Mapping):
			snap_ptr.update(additional_state)

		torch.save(snap_ptr, filepath)
		# Ensure its persisted
		if persist:
			f = open(filepath, 'a+')
			os.fsync(f.fileno())	# 等待
			f.close()

		update_stats(
				filepath,
				iter_chk=iter_chk,
				overwrite=overwrite,
				epoch_chk = epoch_chk,
				linkpath=linkpath)


	def _snapshot_and_persist_async(
		self,
		filepath, 
		active_snapshot,
		in_progress_snapshot,
		lock,
		snap_ptr,
		additional_state=None,
		persist=True,
		linkpath=None,
		iter_chk = None,
		epoch_chk = None,
		overwrite = True,
		profile=False):

		s = time.time()
		print("[{}] START ASYNC".format(time.time()))
		if active_snapshot.value == 1:
			print("ERROR! Snapshot active")
			return

		#with lock: 
		#	in_progress_snapshot.value = 1

		print("In progress snapshot val = {}".format(in_progress_snapshot.value))
		#	snap_ptr = {}
		# DO CPU snapshot	
		snapshot = {}
		for name, ref in snap_ptr.items():
			snapshot[name] = {}
			snapshot[name] = _to_cpu(ref)
		print("Time for CPU snapshot = {}s".format(time.time()-s))
	
		with lock:
			in_progress_snapshot.value = 0
			active_snapshot.value = 1
		print("In progress snapshot val = {}".format(in_progress_snapshot.value))
		print("[{}] START ASYNC PERSIST".format(time.time()))

		if isinstance(additional_state, Mapping):
			snapshot.update(additional_state)

		if profile:
			with lock:
				active_snapshot.value = 0
			return

		torch.save(snapshot, filepath)

		with lock:
			active_snapshot.value = 0

		# Ensure its persisted
		f = open(filepath, 'a+')
		os.fsync(f.fileno())
		f.close()
		
		update_stats(
				filepath,
				iter_chk=iter_chk,
				overwrite=overwrite,
				epoch_chk = epoch_chk,
				linkpath=linkpath)
		print("Time to checkpoint = {}s".format(time.time()-s))
		print("[{}] END ASYNC".format(time.time()))



	"""
	Restores the checkpoint at given path

	Returns the part of checkpoint that was not among
	the tractable items that were registered with CF
	"""

	def _restore( \
		self, \
		filepath='model.chk',
		gpu = 0):
		# map_location=lambda storage, loc: storage.cuda(args.gpu))
		checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage.cuda(gpu))

		# reinitialize state of tractable objects
		for name, ref in self.tracking_map.items():
			try:
				ref.load_state_dict(checkpoint[name])
				del checkpoint[name]
			except ValueError:
				print("Corrupt checkpoint")

		if len(checkpoint.keys()) > 0:	# 说明还有备份数据 未被恢复
			return checkpoint

		return None


def update_stats(
		filepath,
		iter_chk = None,
		epoch_chk=None,
		overwrite = True,
		linkpath=None):		# epoch边界的checkpoint文件保存地址 当不为None时epoch_chk也不为None
	
		# TODO : This incorrectly deletes epochs if old.
		# Manage checkpoints better
		if iter_chk is not None:
			fname = fname_from_path(filepath)
			if fname not in iter_chk:
				iter_chk.append(fname)		# 更新iter_chk

			if overwrite and len(iter_chk) > 1:
				del_filepath = os.path.join(os.path.dirname(filepath), iter_chk[0]+'.chk')
				#print("Deleting {}".format(del_filepath))
				if os.path.exists(del_filepath):
					os.remove(del_filepath)		# 删除前面的checkpoint文件，只保留刚刚append的最新的
				del iter_chk[0]

		if linkpath is not None and epoch_chk is not None:
			epoch_chk.append(fname_from_path(linkpath))



def _to_cpu(ele, snapshot=None):
	#while True:
		if snapshot is None:
				snapshot = {}
		if hasattr(ele, 'cpu'):
				snapshot = ele.cpu()	# .cpu(): 将GPU中的张量复制到CPU
		elif isinstance(ele, dict):
				snapshot = {}
				for k,v in ele.items():
						snapshot[k] = None
						snapshot[k] = _to_cpu(v, snapshot[k])
		elif isinstance(ele, list):
				snapshot  = [None for _ in range(len(ele))]
				for idx, v in enumerate(ele):
						snapshot[idx] = _to_cpu(v, snapshot[idx])

		return snapshot

	


"""
Get the filename without the suffix/prefix
"""
def fname_from_path(path):
	return os.path.splitext(os.path.basename(path))[0]	 # os.path.basename: 返回文件名
