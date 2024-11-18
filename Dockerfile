FROM us-docker.pkg.dev/colab-images/public/runtime
RUN apt update -y
RUN apt install wget
FROM us-docker.pkg.dev/colab-images/public/runtime:release-colab_20230921-060057_RC00
RUN python -m pip -q install git+https://github.com/sokrypton/ColabDesign.git@gamma
RUN apt-get update && apt-get install aria2