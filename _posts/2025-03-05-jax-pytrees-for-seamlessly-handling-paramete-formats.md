---
title: "JAX pytrees for handling parmeter specification formats"
layout: post
mathjax: true
---

***TLDR**; Model parameters can typically be specified in arbitrary formats depending on the model type and the training engineer's choice. This creates a problem for researchers who would like to analyze multiple pre-trained models with varying types / architectures. JAX `pytrees` provide a seamless interface for working with a range of different parameter specification formats without drowning in custom code.*


