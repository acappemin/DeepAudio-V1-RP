FROM r8.im/cog-base:cuda11.8-python3.10

# 安装 Cog CLI（构建包）
COPY .cog/tmp/build*/cog-*.whl /tmp/cog.whl
RUN pip install /tmp/cog.whl

# 拷贝依赖和项目文件
COPY requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY . /code
WORKDIR /code
