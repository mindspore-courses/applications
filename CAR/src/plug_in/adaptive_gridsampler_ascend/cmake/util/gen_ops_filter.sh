# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [[ -z "$1" ]]; then
    echo -e "[ERROR] No source dir provided"
    exit 1
fi

if [[ -z "$2" ]]; then
    echo -e "[ERROR] No destination dir provided"
    exit 1
fi

src=$1
dest_file=$2/npu_supported_ops.json

if [ -f "$dest_file" ];then
    chmod u+w $dest_file
fi

echo $*

add_ops() {
    name=$1
    isHeavy=$2
    file=$3
    grep -w "\"$name\"" ${file} >/dev/null
    if [ $? == 0 ];then
        return
    fi
    echo "  \"${name}\": {" >> ${file}
    echo "    \"isGray\": false," >> ${file}
    echo "    \"isHeavy\": ${isHeavy}" >> ${file}
    echo "  }," >> ${file}
}

echo "{" > ${dest_file}
ini_files=$(find ${src} -name "*.ini")
for file in ${ini_files} ; do
    name=$(grep '^\[' ${file} | sed 's/\[//g' | sed 's/]//g' | sed 's/\r//g')
    grep 'heavyOp.flag' ${file} >/dev/null
    if [ $? == 0 ];then
        isHeavy=$(grep 'heavyOp.flag' ${file} | awk -F= '{print $2}')
    else
        isHeavy="false"
    fi
    add_ops ${name} ${isHeavy} ${dest_file}
done
echo "}" >> ${dest_file}
file_count=$(cat ${dest_file} | wc -l)
line=$(($file_count-1))
sed -i "${line}{s/,//g}" ${dest_file}

chmod 640 "${dest_file}"
echo -e "[INFO] Succeed generated ${dest_file}"

exit 0
