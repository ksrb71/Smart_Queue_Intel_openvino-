#!/usr/bin/env python
# coding: utf-8

# # Smart Queue Monitoring System - Manufacturing Scenario
# 
# ## Overview
# Now that you have your Python script and job submission script, you're ready to request an **IEI Tank-870** edge node and run inference on the different hardware types (CPU, GPU, VPU, FPGA).
# 
# After the inference is completed, the output video and stats files need to be retrieved and stored in the workspace, which can then be viewed within the Jupyter Notebook.
# 
# ## Objectives
# * Submit inference jobs to Intel's DevCloud using the `qsub` command.
# * Retrieve and review the results.
# * After testing, go back to the proposal doc and update your original proposed hardware device.

# ## Step 0: Set Up
# 
# #### IMPORTANT: Set up paths so we can run Dev Cloud utilities
# You *must* run this every time you enter a Workspace session.
# (Tip: select the cell and use **Shift+Enter** to run the cell.)

# In[8]:


get_ipython().run_line_magic('env', 'PATH=/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/intel_devcloud_support')
import os
import sys
sys.path.insert(0, os.path.abspath('/opt/intel_devcloud_support'))
sys.path.insert(0, os.path.abspath('/opt/intel'))


# ### Step 0.1:  (Optional-step): Original Video
# 
# If you are curious to see the input video, run the following cell to view the original video stream we'll be using for inference.

# In[9]:


import videoHtml
videoHtml.videoHTML('Manufacturing', ['original_videos/Manufacturing.mp4'])


# ## Step 1 : Inference on a Video
# 
# In the next few cells, You'll submit your job using the `qsub` command and retrieve the results for each job. Each of the cells below should submit a job to different edge compute nodes.
# 
# The output of the cell is the `JobID` of your job, which you can use to track progress of a job with `liveQStat`.
# 
# You will need to submit a job for each of the following hardware types:
# * **CPU**
# * **GPU**
# * **VPU**
# * **FPGA**
# 
# **Note:** You will have to submit each job one at a time and retrieve their results. 
# 
# After submission, they will go into a queue and run as soon as the requested compute resources become available. 
# (Tip: **shift+enter** will run the cell and automatically move you to the next cell.)
# 
# If your job successfully runs and completes, once you retrieve your results, it should output a video and a stats text file in the `results/manufacturing/<DEVICE>` directory.
# 
# For example, your **CPU** job should output its files in this directory:
# > **results/manufacturing/cpu**
# 
# **Note**: To get the queue labels for the different hardware devices, you can go to [this link](https://devcloud.intel.com/edge/get_started/devcloud/).
# 
# The following arguments should be passed to the job submission script after the `-F` flag:
# * Model path - `/data/models/intel/person-detection-retail-0013/<MODEL PRECISION>/`. You will need to adjust this path based on the model precision being using on the hardware.
# * Device - `CPU`, `GPU`, `MYRIAD`, `HETERO:FPGA,CPU`.
# * Manufacturing video path - `/data/resources/manufacturing.mp4`.
# * Manufacturing queue_param file path - `/data/queue_param/manufacturing.npy`.
# * Output path - `/output/results/manufacturing/<DEVICE>` This should be adjusted based on the device used in the job.
# * Max num of people - This is the max number of people in queue before the system would redirect them to another queue.

# ## Step 1.1: Submit to an Edge Compute Node with an Intel CPU
# In the cell below, write a script to submit a job to an <a 
#     href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
#     Tank* 870-Q170</a> edge node with an <a 
#     href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core™ i5-6500TE processor</a>. The inference workload should run on the CPU.

# In[10]:


#Submit job to the queue
cpu_job_id = get_ipython().getoutput('qsub queue_job.sh -d . -l nodes=1:tank-870:i5-6500te -F "CPU /data/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013 /data/resources/manufacturing.mp4 /data/queue_param/manufacturing.npy /output/results/manufacturing/cpu 2" -N ksrb71_command')
print(cpu_job_id[0])


# #### Check Job Status
# 
# To check on the job that was submitted, use `liveQStat` to check the status of the job.
# 
# Column `S` shows the state of your running jobs.
# 
# For example:
# - If `JOB ID`is in Q state, it is in the queue waiting for available resources.
# - If `JOB ID` is in R state, it is running.

# In[11]:


import liveQStat
liveQStat.liveQStat()


# #### Get Results
# 
# Run the next cell to retrieve your job's results.

# In[12]:


import get_results
get_results.getResults(cpu_job_id[0], filename='output.tgz', blocking=True)


# #### Unpack your output files and view stdout.log

# In[13]:


get_ipython().system('tar zxf output.tgz')


# In[14]:


get_ipython().system('cat stdout.log')


# #### View stderr.log
# This can be used for debugging.

# In[15]:


get_ipython().system('cat stderr.log')


# #### View Output Video
# Run the cell below to view the output video. If inference was successfully run, you should see a video with bounding boxes drawn around each person detected.

# In[16]:


import videoHtml

videoHtml.videoHTML('Manufacturing CPU', ['results/manufacturing/cpu/output_video.mp4'])


# ## Step 1.2: Submit to an Edge Compute Node with CPU and IGPU
# In the cell below, write a script to submit a job to an <a 
#     href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
#     Tank* 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel® Core i5-6500TE</a>. The inference workload should run on the **Intel® HD Graphics 530** integrated GPU.

# In[2]:


#Submit job to the queue
gpu_job_id = get_ipython().getoutput('qsub queue_job.sh -d . -l nodes=1:tank-870:i5-6500te:intel-hd-530 -F "GPU /data/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013 /data/resources/manufacturing.mp4 /data/queue_param/manufacturing.npy /output/results/manufacturing/gpu 2" -N ksrb71_command')
print(gpu_job_id[0])


# ### Check Job Status
# 
# To check on the job that was submitted, use `liveQStat` to check the status of the job.
# 
# Column `S` shows the state of your running jobs.
# 
# For example:
# - If `JOB ID`is in Q state, it is in the queue waiting for available resources.
# - If `JOB ID` is in R state, it is running.

# In[11]:


import liveQStat
liveQStat.liveQStat()


# #### Get Results
# 
# Run the next cell to retrieve your job's results.

# In[12]:


import get_results
get_results.getResults(gpu_job_id[0], filename='output.tgz', blocking=True)


# #### Unpack your output files and view stdout.log

# In[13]:


get_ipython().system('tar zxf output.tgz')


# In[14]:


get_ipython().system('cat stdout.log')


# #### View stderr.log
# This can be used for debugging.

# In[15]:


get_ipython().system('cat stderr.log')


# #### View Output Video
# Run the cell below to view the output video. If inference was successfully run, you should see a video with bounding boxes drawn around each person detected.

# In[1]:


import videoHtml

videoHtml.videoHTML('Manufacturing GPU', ['results/manufacturing/gpu/output_video.mp4'])


# ## Step 1.3: Submit to an Edge Compute Node with a Neural Compute Stick 2
# In the cell below, write a script to submit a job to an <a 
#     href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
#     Tank 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core i5-6500te CPU</a>. The inference workload should run on an <a 
#     href="https://software.intel.com/en-us/neural-compute-stick">Intel Neural Compute Stick 2</a> installed in this  node.

# In[17]:


#Submit job to the queue
vpu_job_id = get_ipython().getoutput('qsub queue_job.sh -d . -l nodes=1:tank-870:i5-6500te:intel-ncs2 -F "MYRIAD /data/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013 /data/resources/manufacturing.mp4 /data/queue_param/manufacturing.npy /output/results/manufacturing/vpu 2" -N ksrb71_command')
print(vpu_job_id[0])


# ### Check Job Status
# 
# To check on the job that was submitted, use `liveQStat` to check the status of the job.
# 
# Column `S` shows the state of your running jobs.
# 
# For example:
# - If `JOB ID`is in Q state, it is in the queue waiting for available resources.
# - If `JOB ID` is in R state, it is running.

# In[18]:


import liveQStat
liveQStat.liveQStat()


# #### Get Results
# 
# Run the next cell to retrieve your job's results.

# In[19]:


import get_results
get_results.getResults(vpu_job_id[0], filename='output.tgz', blocking=True)


# #### Unpack your output files and view stdout.log

# In[20]:


get_ipython().system('tar zxf output.tgz')


# In[21]:


get_ipython().system('cat stdout.log')


# #### View stderr.log
# This can be used for debugging.

# In[22]:


get_ipython().system('cat stderr.log')


# #### View Output Video
# Run the cell below to view the output video. If inference was successfully run, you should see a video with bounding boxes drawn around each person detected.

# In[23]:


import videoHtml

videoHtml.videoHTML('Manufacturing VPU', ['results/manufacturing/vpu/output_video.mp4'])


# ## Step 1.4: Submit to an Edge Compute Node with IEI Mustang-F100-A10
# In the cell below, write a script to submit a job to an <a 
#     href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
#     Tank 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core™ i5-6500te CPU</a> . The inference workload will run on the <a href="https://www.ieiworld.com/mustang-f100/en/"> IEI Mustang-F100-A10 </a> FPGA card installed in this node.

# In[24]:


#Submit job to the queue
fpga_job_id = get_ipython().getoutput('qsub queue_job.sh -d . -l nodes=1:tank-870:i5-6500te:iei-mustang-f100-a10 -F "HETERO:FPGA,CPU /data/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013 /data/resources/manufacturing.mp4 /data/queue_param/manufacturing.npy /output/results/manufacturing/fpga 2" -N ksrb71_command')
print(fpga_job_id[0])


# ### Check Job Status
# 
# To check on the job that was submitted, use `liveQStat` to check the status of the job.
# 
# Column `S` shows the state of your running jobs.
# 
# For example:
# - If `JOB ID`is in Q state, it is in the queue waiting for available resources.
# - If `JOB ID` is in R state, it is running.

# In[25]:


import liveQStat
liveQStat.liveQStat()


# #### Get Results
# 
# Run the next cell to retrieve your job's results.

# In[26]:


import get_results

get_results.getResults(fpga_job_id[0], filename='output.tgz', blocking=True)


# #### Unpack your output files and view stdout.log

# In[27]:


get_ipython().system('tar zxf output.tgz')


# In[28]:


get_ipython().system('cat stdout.log')


# #### View stderr.log
# This can be used for debugging.

# In[29]:


get_ipython().system('cat stderr.log')


# #### View Output Video
# Run the cell below to view the output video. If inference was successfully run, you should see a video with bounding boxes drawn around each person detected.

# In[30]:


import videoHtml

videoHtml.videoHTML('Manufacturing FPGA', ['results/manufacturing/fpga/output_video.mp4'])


# ***Wait!***
# 
# Please wait for all the inference jobs and video rendering to complete before proceeding to the next step.
# 
# ## Step 2: Assess Performance
# 
# Run the cells below to compare the performance across all 4 devices. The following timings for the model are being compared across all 4 devices:
# 
# - Model Loading Time
# - Average Inference Time
# - FPS

# In[31]:


import matplotlib.pyplot as plt

device_list=['cpu', 'gpu', 'fpga', 'vpu']
inference_time=[]
fps=[]
model_load_time=[]

for device in device_list:
    with open('results/manufacturing/'+device+'/stats.txt', 'r') as f:
        inference_time.append(float(f.readline().split("\n")[0]))
        fps.append(float(f.readline().split("\n")[0]))
        model_load_time.append(float(f.readline().split("\n")[0]))


# In[32]:


plt.bar(device_list, inference_time)
plt.xlabel("Device Used")
plt.ylabel("Total Inference Time in Seconds")
plt.show()


# In[33]:


plt.bar(device_list, fps)
plt.xlabel("Device Used")
plt.ylabel("Frames per Second")
plt.show()


# In[34]:


plt.bar(device_list, model_load_time)
plt.xlabel("Device Used")
plt.ylabel("Model Loading Time in Seconds")
plt.show()


# # Step 3: Update Proposal Document
# 
# Now that you've completed your hardware testing, you should go back to the proposal document and validate or update your originally proposed hardware. Once you've updated your proposal, you can move on to the next scenario.
