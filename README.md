# IPGPhormer: Interpretable Pathology Graph-Transformer for Survival Analysis


Pathological images play an essential role in cancer prognosis, while survival analysis, which integrates computational techniques, can predict critical clinical events such as patient mortality or disease recurrence from whole-slide images (WSIs). 
Recent advancements in multiple instance learning have significantly improved the efficiency of survival analysis. However, existing methods often struggle to balance the modeling of long-range spatial relationships with local contextual dependencies and typically lack inherent interpretability, limiting their clinical utility.
To address these challenges, we propose the Interpretable Pathology Graph-Transformer (IPGPhormer), a novel framework that captures the characteristics of the tumor microenvironment and models their spatial dependencies across the tissue. IPGPhormer uniquely provides interpretability at both tissue and cellular levels without requiring post-hoc manual annotations, enabling detailed analyses of individual WSIs and cross-cohort assessments. 
Comprehensive evaluations on four public benchmark datasets demonstrate that IPGPhormer outperforms state-of-the-art methods in both predictive accuracy and interpretability. In summary, our method, IPGPhormer
offers a promising tool for cancer prognosis assessment, paving the way for more reliable and interpretable decision-support systems in pathology.

&nbsp;

![image](https://github.com/35tang/IPGPhormer/blob/main/Figs/framework.png)
Overview of the proposed IPGPhormer architecture. 
First, we use HoverNet to create both tissue graphs and cell graphs, and then extract patch features using CTransPath.
The Patch-Level Transfer Module leverages GAT for local spatial awareness, while the Region-Level Feature Transfer Module converts the graph data into a sequential embedding format and feeds it into the Transformer blocks to capture long-range dependencies. 
Finally, the patch risk score establishes a link between model outputs and both tissue-level and cell-level interpretability.

&nbsp;

## Interpretability
![image](https://github.com/35tang/IPGPhormer/blob/main/Figs/tissue.png)
The tissue-interpretability approach shows patch-wise risk values aligned with pathology expert assessments.

&nbsp;

![image](https://github.com/35tang/IPGPhormer/blob/main/Figss/cell.png)
Cross-cohort cell-interpretability analysis aids in the identification of potential biomarkers.

&nbsp;

## Updates
The code repository is being updated


&nbsp;

## Tips
The relevant codes for extracting cell statistical features are referenced from [SI-MIL](https://github.com/bmi-imaginelab/SI-MIL).

The relevant codes for graph construction are referenced from [WSI-HGNN](https://github.com/HKU-MedAI/WSI-HGNN).
