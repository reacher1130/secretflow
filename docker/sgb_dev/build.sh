#!/bin/bash
# Copyright 2024 Ant Group Co., Ltd.
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


set -ex

show_help() {
    echo "Usage: bash build.sh [OPTION]... -v {version}"
    echo "  -v  --version"
    echo "          the version to build with."
    echo "  -r  --reg"
    echo "          docker reg to upload."
    echo "  -l --latest"
    echo "          tag this version as latest."
}

if [[ "$#" -lt 2 ]]; then
    show_help
    exit
fi

DOCKER_REG="secretflow-registry.cn-hangzhou.cr.aliyuncs.com/secretflow"

while [[ "$#" -ge 1 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift
            if [[ "$#" -eq 0 ]]; then
                echo "Version shall not be empty."
                echo ""
                show_help
                exit 1
            fi
            shift
        ;;
        -r|--reg)
            DOCKER_REG="$2"
            shift
            if [[ "$#" -eq 0 ]]; then
                echo "Docker reg shall not be empty."
                echo ""
                show_help
                exit 1
            fi
            shift
        ;;
        -l|--latest)
            LATEST=1
            shift
        ;;
        *)
            echo "Unknown argument passed: $1"
            exit 1
        ;;
    esac
done


if [[ -z ${VERSION} ]]; then 
    echo "Please specify the version."
    exit 1
fi

GREEN="\033[32m"
NO_COLOR="\033[0m"

IMAGE_NAME=sgb:${VERSION}
IMAGE_TAG=${DOCKER_REG}/${IMAGE_NAME}
echo -e "Building ${GREEN}${IMAGE_TAG}${NO_COLOR}"
(cd ../.. && rm -rf dist/ build/)

docker run -it --rm -e SF_BUILD_DOCKER_NAME=${IMAGE_NAME} --mount type=bind,source="$(pwd)/../../../secretflow",target=/home/admin/src -w /home/admin --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=NET_ADMIN --privileged=true secretflow/release-ci:latest /home/admin/src/docker/sgb_dev/entry.sh
(cd ../ && cp -r release/.nsjail sgb_dev/ && cp release/.condarc sgb_dev/)
docker build . -f Dockerfile -t ${IMAGE_TAG} --build-arg BUILD_TIMEOUT=1200
echo -e "Finish building ${GREEN}${IMAGE_TAG}${NO_COLOR}"
rm -rf .nsjail
rm -f .condarc
rm -f *.whl
