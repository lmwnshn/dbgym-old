#!/usr/bin/env bash

if docker volume inspect dsb_queries; then
  echo "Volume 'dsb_queries' exists."
  exit
fi

echo "Volume 'dsb_queries' doesn't exist, creating."

set -euxo pipefail

ROOT_DIR=$(pwd)

if [ ! -d "./build/dsb" ]; then
  mkdir -p ./build
  cd ./build
  git clone git@github.com:lmwnshn/dsb.git --single-branch --branch fix_ubuntu --depth 1
  cd ./dsb/code/tools
  make
  cd "${ROOT_DIR}"
fi

function generate_queries() {
  num_streams=$1
  output_dir=$2

  cd "${ROOT_DIR}/build/dsb/code/tools"
  set +x
  mkdir -p generated_queries
  for template_path in ../../query_templates_pg/*/query*.tpl ; do
    folder=$(dirname "${template_path}")
    template=$(basename "${template_path}")

    rm -rf ./tmp
    mkdir -p ./tmp
    ./dsqgen -streams "${num_streams}" -output_dir ./tmp -dialect postgres -directory "${folder}" -template "${template}"

    cd ./tmp/
    for f in * ; do
      name=$(echo "${template}_${f}" | sed "s/.tpl_query_/-/")
      mv "${f}" "${name}"
    done
    cd ../

    mv ./tmp/* "${output_dir}"
    rm -rf ./tmp
  done
  cd "${ROOT_DIR}"

  set -x
}

for seed in $(seq 15721 15722) ; do
  cd "${ROOT_DIR}/build/dsb/code/tools/"
  mkdir -p ./generated_queries/default/"${seed}"
  ./distcomp -i ./tpcds.dst -o ./tpcds.idx -rngseed "${seed}" -param_dist default # -param_dist normal -param_sigma 2 -param_center 0
  generate_queries 100 ./generated_queries/default/"${seed}"
  cd "${ROOT_DIR}"
done

docker run --volume=dsb_queries:/dsb_queries --name dsb_queries busybox true
docker cp --quiet ./build/dsb/code/tools/generated_queries/. dsb_queries:/dsb_queries
docker rm dsb_queries
