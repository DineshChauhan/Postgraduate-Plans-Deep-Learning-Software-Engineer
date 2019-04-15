# Postgraduate Machine Learning Plan

## Deep Learning Software Engineering Focus

* FastAI Part 2 Lectures (Watch, Listen, Share Lecture Notes) -- Focus is on *Deep Learning from the Foundations*, *State of the Art Computer Vision Techniques*, *Deep Learning for Audio*, and *Swift for Tensorflow framework*
* FastAI [Homemade Homework Assignments](https://github.com/dcooper01/FastAI-Deep-Learning-2019-Part-2-Resources) (Implement Library from Scratch + Learn PyTorch)
* Learn Swift for TensorFlow Basics (Chris Lattner lectures)
* Contribute to open-source libraries (FastAI Audio & Swift for TensorFlow)
* Practical Projects (Focused on Computer Vision):
  * [MNIST](http://yann.lecun.com/exdb/mnist/), [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), and [Imagewoof](https://github.com/fastai/imagenette) *Classification using software library written from scratch (with PyTorch for AutoDiff) -- write matrix operations, dataloader, optimizer, etc. from scratch in pure python*
  * [Aerial Cactus Identification](https://www.kaggle.com/c/aerial-cactus-identification/data) *(Classification)*
  * [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) *(Classification)*
  * [Knee MRI](https://stanfordmlgroup.github.io/competitions/mrnet/) *(Classification)*
  * [NOAA Fisheries Steller Sea Lion Population Count](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/data) *(Object Detection + Maybe YOLO/YOLACT implementation)*
  * [Airbus Ship Detection](https://www.kaggle.com/c/airbus-ship-detection/overview) *(Segmentation + Maybe Super Resolution GAN)*
  * [DSTL Satellite Imagery Feature Detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection) *(Segmentation + Maybe Super Resolution GAN)* (Consider using *Transfer Learning* from [EuroSat Data](https://paperswithcode.com/paper/eurosat-a-novel-dataset-and-deep-learning))
  * [DeNoising Dirty Documents with GANs](https://www.kaggle.com/c/denoising-dirty-documents) *(Image Generation with GAN)*
  * More Complex Medical Segmentation Project
    * Possibilities include [Alzheimers/Dementia Classification with MRI Data](https://caddementia.grand-challenge.org/), [Kidney Segmentation](https://kits19.grand-challenge.org/home/), [Pneumonia Chest X-Ray Competition](https://www.kaggle.com/parthachakraborty/pneumonia-chest-x-ray), etc.

## Natural Language Processing Focus

* [NLP CS-224n Stanford Lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
* [NLP CS-224n Stanford Assignments](http://web.stanford.edu/class/cs224n/)
* Work Projects:
  * Deep Learning for Search and Match between job postings and people/resume data (Utilizes GANs, Attention Networks, etc.)
* Personal Projects (List to be edited during CS-224n lecture progression):
  * [Rotten Tomatoes](https://www.reddit.com/r/MachineLearning/comments/b5idqk/p_dataset_480000_rotten_tomatoes_reviews_for_nlp/) Fresh or Not *(Classification / Regression)*
  * Onion Classifier: Predict if a news article is from [r/theonion](https://www.reddit.com/r/theonion) or from [r/nottheonion](https://www.reddit.com/r/nottheonion) *(Classification)*
  * Florida Man Generator: Generate news article headings for crazy [Florida Man](http://reddit.com/r/FloridaMan) stuff + TweetBot *(Text Generation using GPT-2 Fine-Tuning)*
  * Support Case Routing Engine: Simulating support case network and automatically routing tweets/e-mails to the right department *(Classification + Simulation Optimization)*
  * Reddit Censor: Predict which posts will get removed by moderators in real time (choose some contentious sub-reddit, like one affiliated with a political group during US election) *(Classification)*
  * Build a ChatBot for personal website to help generate leads inquiring about freelance projects (i.e. What kind of project are you interested in, when would you like to talk with Daniel (integration to scheduler), etc.)

## Audio Focus

* FastAI Lectures (when released)
* Projects:
  * Compete in Kaggle Competition: [Freesound Audio Tagging](https://www.kaggle.com/c/freesound-audio-tagging)
  * Implement [GANSynth: Adversarial Neural Audio Synthesis](https://paperswithcode.com/paper/gansynth-adversarial-neural-audio-synthesis) or [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

## Misc

* Consider creating a basic genetic algorithm library in Swift / Swift for TF
* Explore scheduling Genetic Algorithm hyperparameters (i.e. annealing mutation rate, etc.)
* Implement one of Ken Stanley's papers on NeuroEvolution or Novelty Search