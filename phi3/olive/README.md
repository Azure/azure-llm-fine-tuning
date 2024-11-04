---
layout: default
title: Optimization using Microsoft Olive
permalink: /2_2_optimization/
parent: Lab 2. SLM/LLM Fine-tuning on Azure ML Studio
nav_order: 5.1
---

# Optimize SLM using Microsoft Olive

This hands-on considers on-device or hybrid deployment scenarios.

### Overview

Microsoft Olive is a hardware-aware AI model optimization toolchain developed by Microsoft to streamline the deployment of AI models. Olive simplifies the process of preparing AI models for deployment by making them faster and more efficient, particularly for use on edge devices, cloud, and various hardware configurations. It works by automatically applying optimizations to the AI models, such as reducing model size, lowering latency, and improving performance, without requiring manual intervention from developers.

Key features of Microsoft Olive include:

- Automated optimization: Olive analyzes and applies optimizations specific to the model’s hardware environment.
- Cross-platform compatibility: It supports various platforms such as Windows, Linux, and different hardware architectures, including CPUs, GPUs, and specialized AI accelerators.
- Integration with Microsoft tools: Olive is designed to work seamlessly with Microsoft AI services like Azure, making it easier to deploy optimized models in cloud-based solutions.
