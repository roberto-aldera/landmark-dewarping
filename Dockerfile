FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN pip install pytorch_lightning \
				pandas \
				matplotlib
