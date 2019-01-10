# Copyright 2019 Babylon Partners. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This source code is derived from SentEval source code.
# SentEval Copyright (c) 2017-present, Facebook, Inc.
# ==============================================================================

#
# Download and tokenize data with MOSES tokenizer
#

data_path=.
preprocess_exec=./tokenizer.sed

# Get MOSES
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
SCRIPTS=mosesdecoder/scripts
MTOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LOWER=$SCRIPTS/tokenizer/lowercase.perl

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

PTBTOKENIZER="sed -f tokenizer.sed"

mkdir $data_path


# STS 2012, 2013, 2014, 2015, 2016
declare -A STS_tasks
declare -A STS_paths
declare -A STS_subdirs

STS_tasks=(["STS12"]="MSRpar MSRvid SMTeuroparl surprise.OnWN surprise.SMTnews" ["STS13"]="FNWN headlines OnWN" ["STS14"]="deft-forum deft-news headlines OnWN images tweet-news" ["STS15"]="answers-forums answers-students belief headlines images" ["STS16"]="answer-answer headlines plagiarism postediting question-question")

STS_paths=(["STS12"]="http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip" ["STS13"]="http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip" ["STS14"]="http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip" ["STS15"]="http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip"
["STS16"]="http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip")

STS_subdirs=(["STS12"]="test-gold" ["STS13"]="test-gs" ["STS14"]="sts-en-test-gs-2014" ["STS15"]="test_evaluation_task2a" ["STS16"]="sts2016-english-with-gs-v1.0")


### STS datasets

# STS12, STS13, STS14, STS15, STS16
mkdir $data_path/STS

for task in "${!STS_tasks[@]}"; #"${!STS_tasks[@]}";
do
    fpath=${STS_paths[$task]}
    echo $fpath
    curl -Lo $data_path/STS/data_$task.zip $fpath
    unzip $data_path/STS/data_$task.zip -d $data_path/STS
    mv $data_path/STS/${STS_subdirs[$task]} $data_path/STS/$task-en-test
    rm $data_path/STS/data_$task.zip

    for sts_task in ${STS_tasks[$task]}
    do
        fname=STS.input.$sts_task.txt
        task_path=$data_path/STS/$task-en-test/

        if [ "$task" = "STS16" ] ; then
            echo 'Handling STS2016'
            mv $task_path/STS2016.input.$sts_task.txt $task_path/$fname
            mv $task_path/STS2016.gs.$sts_task.txt $task_path/STS.gs.$sts_task.txt
        fi



        cut -f1 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp1
        cut -f2 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp2
        paste $task_path/tmp1 $task_path/tmp2 > $task_path/$fname
        rm $task_path/tmp1 $task_path/tmp2
    done

done

# remove moses folder
rm -rf mosesdecoder
