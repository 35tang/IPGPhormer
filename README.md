# IPGPhormer: Interpretable Pathology Graph-Transformer for Survival Analysis
Pathological images play an essential role in cancer prognosis, while survival analysis, which integrates computational techniques, can predict critical clinical events such as patient mortality or disease recurrence from whole-slide images (WSIs). 
Recent advancements in multiple instance learning have significantly improved the efficiency of survival analysis. However, existing methods often struggle to balance the modeling of long-range spatial relationships with local contextual dependencies and typically lack inherent interpretability, limiting their clinical utility.
To address these challenges, we propose the Interpretable Pathology Graph-Transformer (IPGPhormer), a novel framework that captures the characteristics of the tumor microenvironment and models their spatial dependencies across the tissue. IPGPhormer uniquely provides interpretability at both tissue and cellular levels without requiring post-hoc manual annotations, enabling detailed analyses of individual WSIs and cross-cohort assessments. 
Comprehensive evaluations on four public benchmark datasets demonstrate that IPGPhormer outperforms state-of-the-art methods in both predictive accuracy and interpretability. In summary, our method, IPGPhormer
offers a promising tool for cancer prognosis assessment, paving the way for more reliable and interpretable decision-support systems in pathology.


![image](https://github.com/35tang/IPGPhormer/blob/main/framework.png)
