
songci=/path/to/ci
ttf_file=/path/to/chinese_scripts/regular_script/STXINGKA.TTF

tstci=/path/to/ci/ci.song.10000.json

#songci=/disk2/pwj/workspace/data/ci
#ttf_file=/disk2/pwj/workspace/data/chinese_scripts/regular_script/STXINGKA.TTF
#tstci=/disk2/pwj/workspace/data/ci/ci.song.10000.json


if [ ! -d $songci ]; then
    echo "Song Ci data not found!"
    exit 0
fi

if [ ! -f $ttf_file ]; then
    echo "TTF file not found!"
    exit 0
fi

mkdata=./data_helper.py
mkvec=./gen_word_embedding.py
doc2vec=./doc2vec.py

mkdir -p ../data
mkdir -p ../md

echo "Stage: Data Preparation... (Author: Peng Wenjie)"

python $mkdata $songci $ttf_file || exit 1
    
echo "Char2vec Model Training... (Author: Peng Wenjie)"

python main.py || exit 1;
python $mkvec || exit 1;

echo "Make Char2vec for Cluster... (Author: Peng Wenjie)"

python $doc2vec $tstci || exit 1;

echo "K-Means cluster... (Author: Li Zhu)"
vec1=../data/char2vec_raw.pickle
vec2=../data/char2vec_with_glyce.pickle

kpy=./kmeans_lz.py
python $kpy $vec1 || exit 1
python $kpy $vec2 || exit 1

#exit 0

echo "Hierarchical Cluster... (Author: Zhang Yuqing)"

hpy=./hclust_zyq.py
python $hpy $vec1 || exit 1
python $hpy $vec2 || exit 1

echo "Done."
# To visulize the cluster results, Copy the json results generated from
# cluster algorithms to https://observablehq.com/@d3/cluster-dendrogram
# 
