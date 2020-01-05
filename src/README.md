基于图像信息字向量的宋词聚类  

小组成员：  
    彭文杰  
    李柱  
    张宇晴  

小组分工:
（1）彭文杰：数据处理、字符向量模型  
（2）李柱： k-means聚类、D3视觉化聚类结果  
（3）张宇晴： 层次聚类、D3视觉化聚类结果  


（1）数据处理、字符向量模型@彭文杰    
    运行脚本前，需下载以下文件： 

    （a）宋词数据集（https://github.com/chinese-poetry/chinese-poetry） 
    （b）ttf文件（https://github.com/ShannonAI/glyce/blob/master/glyce/fonts/README.md） 

    并在data_helper.py中配置正确的路径  
    生成数据文件后，运行main.py训练字符向量  

（2）k-means聚类、D3视觉化@李柱  
    配置正确的路径后运行kmeans_lz.py，生成json文件，并将其上传到https://observablehq.com/@d3/cluster-dendrogram,进行视觉化  
（3）层次聚类、D3视觉化@张宇晴  
    执行hcluter_zyq.py，其他同上  

一行命令执行：  
bash run.sh
（需要在run.sh里面配置好路径）
