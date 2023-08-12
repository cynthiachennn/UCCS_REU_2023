# UCCS REU 2023

Work in progress codebase for project on Classifying Individual Finger Movements for Brain Computer Interfaces.
Current state of results:
56% average classification accuracy on individual subjects, barely beating state of the art.
55% average classification accuracy with transfer learning, this is novel.

In order to reproduce my results:
- Download all 5F experiemental paragdim files from https://doi.org/10.6084/m9.figshare.c.3917698.v1
- Run reformatting_200 and reformatting_1000 in Matlab 2023a to get event files
- Run raw_data.ipynb to get numpy files of raw data, which is the processed input for the final model
- Run transformer.py, which is the best classifier i have so far for single subject classification
- Run transfer.py, which is the best classifier for transfer learning
- Other files are other models or preprocessing pipelines that were not as effective
