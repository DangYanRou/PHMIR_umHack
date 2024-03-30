# FinGuru Streamlit App Installation Guide

This guide will walk you through the steps to set up and run FinGuru Streamlit App on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Anaconda](https://www.anaconda.com/products/distribution)
- [Git](https://git-scm.com/)

## Step 1: Create a Virtual Environment

Open Anaconda Prompt and create a new virtual environment by running the following command:

```bash
conda create --name UMHack python=3.12.2
```

Activate the virtual environment:

```bash
conda activate UMHack
```


## Step 2: Clone the Repository

Create a folder where you want to store your project files. Then, navigate to the folder using the Anaconda Prompt and clone your Git repository by running the following command:

```bash
git clone https://github.com/DangYanRou/PHMIR_umHack.git
```


## Step 3: Install Dependencies

Navigate into the cloned repository directory:

```bash
cd <repository_folder>
```

Install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## Step 4: Run the Streamlit Program

After installing the dependencies, you can now run FinGuru Streamlit App. Make sure you are still in the repository directory and in the same virtual environment.

Run the following command:

```bash
streamlit run Home.py
```


This command will start the Streamlit server and open your default web browser with the Streamlit application running. You can now interact with the application.

## Conclusion

You have successfully set up and run a Streamlit Python program on your local machine. Happy Streamlit-ing!

