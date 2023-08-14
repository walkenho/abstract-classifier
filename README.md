# Multilabel Classification of Journal and Conference Abstracts

## Overview

Text classification is a common example of an application for supervised machine learning on texts.

Example use-cases include:

* Detecting topics of abstracts submitted to conferences/for publication in scientific journals. 
* Detecting types of toxic comments in internet forums as is done in the [Toxic Comment Classification Challenge](https://paperswithcode.com/task/toxic-comment-classification).
* Automating routing of customer queries in order to scale customer support efforts.  

The goal of this project is to build ML models that can predict the topics of an abstract that a user is submitting to a conference. The task at hand is framed as a multilabel classification task, meaning that each abstract can have multiple tags associated with it. Multilabel classification tasks are typically solved in one of the three following ways:

* Reframe the task into a multiclass classification task, where each class represents a combination of original labels. This scales very poorly with the number of labels used. 
* Reframe the task of predicting N multi-labels into N single-label classification tasks.
* Reframe the task of predicting N multi-labels into N single-label classification tasks, but chain them together, so that each classifier receives as additional inputs the outputs of the previous classifiers.  

In this project, I will be using the second approach - reframing the task into independent single-label classification tasks.

Since we are dealing with imbalanced data (each label occurs in fewer than 10% of articles) and I am interested in both capturing many abstract labels as well as not producing too many false positives, I choose to assess model performance using the f1 score and then average the f1-score across classes using the weighted average, since I consider it more important to correctly predict as many tags as possible than to get equally good performance across all labels.

The project is developed in a combination of source code as well as notebooks for exploration. Figures can be found in reports/figures.

## Usage

This project comes with a Makefile to create the relevant datasets:

* `make arxiv-data`: Scrape arXiv data for relevant categories.
* `make interim-data`: Create raw dataset from arXiv articles.
* `make processed-data-tfidf`: Clean data for use in tdfidf model.
* `make processed-data-distilbert`: Clean data for use in distilbert model.

Models are currently built and trained in the respective notebooks. This is supposed to move into code in the future.

It also comes with tooling for code quality. Check the makefile for details.

## Project Results

For a complete summary of the project results (so far), see the notebooks/00-Readme_and_Report.ipynb.
