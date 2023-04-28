import sys
import os
import subprocess

str_bw_file = "./.STR_BW"

def get_storage_bandwidth(disk="/datadrive/mnt2"): 
    str_bw = strProfileExists() 
    if str_bw is not None:
        return str_bw
    else:
        paths = disk.split('/')   #[ ,'datadrive','mnt2']
        print(paths)
        mnt_paths = [s for s in paths if s.startswith("mnt")]
        print(mnt_paths)
        disk = mnt_paths[0]  # 'mnt2'
        dev_cmd = ['grep', disk, '/proc/mounts']     #查找/proc/mounts目录下含有内容disk的文件  
        dev_cmd_cut = ['cut', '-d', ' ', '-f', '1']     #对文件的每一行，以空格作为分隔符打印第一个字段
        p = subprocess.Popen(dev_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)   
        #初始化subprocess.popen类，创建一个子进程   subprocess.PIPE 表示为子进程创建新的管道
        output = subprocess.check_output(dev_cmd_cut, stdin=p.stdout) 
        #执行一个外部命令并获取输出 
        p.wait()
        print("Output = {}".format(output))
        if p.returncode != 0: 
            out, err = p.communicate()
            print("Error : {}".format(err.decode('utf-8'))) 
            return 0,0  
        device = output.decode('utf-8').rstrip()    #删除字符串末尾的空白符
        print("Measuring bandwidth of storage dev  {}".format(device))
        dev_bw = ['hdparm', '-t', device]   #linux 命令：获取读/写磁盘速度
        #dev_bw = ['sudo', 'hdparm', '-t', device] 
        p = subprocess.Popen(dev_bw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()  
        result = out.decode('utf-8') 
        print(result, err.decode('utf-8'))
        str_bw = result.split()[-2]   #默认分隔符：空格   
        os.environ['STR_BW'] = str_bw 
        with open(str_bw_file, 'w+') as wf: 
            wf.write(str_bw) 
        return str_bw

def strProfileExists():
    if 'STR_BW' in  os.environ:   # os.environ：获取环境变量键值对
        str_bw = os.environ['STR_BW']   #str_bw：'STR_BW'对应的目录
        return float(str_bw)    #存储的读写速度是数字字符串，转换为float
    elif os.path.exists(str_bw_file):
        with open(str_bw_file, 'r') as rf:
            str_bw = rf.readline()   #读取一行数据
        return float(str_bw)   
    else:
        return None
