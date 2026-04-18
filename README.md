# Translating System Logs to Incident Reports 

An automated ML pipeline built with TensorFlow and Keras that uses a custom Transformer model to translate raw, cryptic system logs into clear, actionable, and human-readable incident reports.

## Overview

System administrators and SREs often have to parse through dense server logs to figure out what caused an outage. This project demonstrates a Sequence-to-Sequence (Seq2Seq) approach using a Transformer architecture to read log patterns (like CPU saturation, Memory exhaustion, or Disk I/O bottlenecks) and generate natural language summaries.

## Features

* **Synthetic Dataset Generator:** Automatically generates training data containing realistic log sequences and their corresponding incident reports.
* **Custom Transformer Architecture:** Implements a from-scratch Transformer model, including Positional Encoding and a Custom Multi-Head Attention layer.
* **Robust Preprocessing:** Handles vocabulary tokenization, log normalization (masking IP addresses, timestamps, and numeric values), and sequence padding.
* **Model Serialization:** Saves and loads custom Keras objects seamlessly for reuse without retraining.

## Prerequisites

Make sure you have Python 3.x installed along with the following libraries:

* `tensorflow` 
* `numpy`

You can install the required dependencies using:
```bash
pip install tensorflow numpy
