# **Summary of Quantitative Evaluation Metrics**

This document provides a condensed overview of the quantitative metrics used to evaluate the generative lighting system.

### **I. Intention-Based Structural and Temporal Analysis**

This analysis measures the internal coherence and musical alignment of a generated light show without reference to a ground truth.

#### **Structural Correspondence Metrics**

* **SSM Correlation (Γ\_structure)**: Measures the high-level structural similarity between the music and the light show (e.g., verse/chorus structure) by correlating their respective Self-Similarity Matrices (SSMs).  

- Note to myself: Let's use this metric as it is

* **Novelty Correlation (Γ\_novelty)**: Quantifies how well significant transitions in the lighting align with structural changes in the music by correlating their novelty curves.  

- Note to myself: Okay, this one is a bit problematic because this shows the distribution correlation and this is not what we want to use. Because we said that we use something different, maybe the approach that we are using for the ground truth comparison also. Because you see it in the plots itself that the generative things are really, really reacting and setting novelties. But the score is really poor and maybe we have to rethink this score because with 8% it's not good to show. And I think we are just in comparison to the paradigm shift and everything I mentioned in the rigmes and in the markdown files of the repo. I think we have to recalculate this so that we get more comparison of the quality and not of the distribution, I think. 

* **Boundary F-Score (Γ\_boundary)**: Assesses the accuracy of aligning detected structural boundaries (peaks in the novelty curves) between the audio and the lighting.

- Note to myself: I want to kick this, I don't want to use this. 

#### **Rhythmic and Temporal Alignment Metrics**

* **Onset ↔ Change (Γ\_change)**: Measures the low-level synchronicity between musical onsets and any corresponding change in the lighting parameters.  
- Note to myself: I want to use this. 
* **Beat ↔ Peak (Γ\_beat↔peak)**: Evaluates how precisely lighting intensity peaks align with the main musical beat in rhythmic sections.  
- Note to myself: I want to use this. 
* **Beat ↔ Valley (Γ\_beat↔valley)**: Similar to peak alignment, this measures how well lighting intensity minima (valleys) align with the musical beat.
- Note to myself: I want to use this 

#### **Dynamic and Color Variation Metrics**
- Note to myself: I need to find a way to explain the percentages over 100% better for these metrics because otherwise it feels like I have cheated and I don't want to seem like a cheater or a liar or something like that because it seems a bit weird to have over 100%. I know where it comes from but I need some more description to that. 
* **RMS ↔ Brightness (Γ\_loud↔bright)**: Measures the correlation between the audio's loudness (RMS energy) and the overall brightness of the lighting.  
- Note to myself: I want to use this. 
* **Intensity Variance (Ψ\_intensity)**: Quantifies the overall dynamic range used in the lighting intensity across the entire sequence.  
- Note to myself: I want to use this. 
* **Color Variance (Ψ\_color)**: Measures the degree of hue and saturation variation, indicating how chromatically dynamic the show is.
- Note to myself: I don't want to use this. 

### **II. Intention-Based Ground Truth Comparison**
- Note to myself: I want to use all of these. I am fine with achievement_ratios.png and the quality_breakdown.png. I don't need a dashboard for this.
This analysis assesses the functional quality of the generated output by comparing its performance on key metrics against human-designed ground truth data.

* **Beat Alignment Ratio**: Compares the beat alignment score of the generated show to the ground truth show. A ratio \>100% indicates superior performance.  
* **Onset Correlation Ratio**: Compares the onset-to-change correlation of the generated show to the ground truth, measuring relative responsiveness.  
* **Structural Similarity Preservation**: Compares the SSM Correlation of the generated show to the ground truth to assess if high-level structure is preserved.  
* **Overall Quality Score**: A single, weighted score that aggregates the above ratios to provide a summarized measure of performance relative to the ground truth.


### **III. Segment-Based Hybrid Oscillator Evaluation**

This analysis evaluates the discrete, high-level decisions made by the oscillator-based model, focusing on the appropriateness of the chosen wave type for each musical segment.

* **Consistency**: Measures the stability and uniformity of the chosen wave type (e.g., 'sine', 'square') within a coherent musical section.  
- Note to myself: I can not find this as a plot??? But I want to use this!!!
* **Musical Coherence**: Evaluates whether the complexity of the selected wave type is appropriate for the musical energy and dynamics of the corresponding segment.  
- Note to myself: I can not find this as a plot??? But I want to use this!!!
* **Transition Smoothness**: Assesses the quality of transitions between different wave types across segment boundaries, penalizing abrupt or musically jarring shifts.  
- Note to myself: I can not find this as a plot??? But I want to use this!!!
* **Distribution Match**: Compares the overall distribution of generated wave types against the distribution found in the human-designed data to identify systemic biases.
- Note to myself: I want to use this. 


### ** Additional Notes about generated plots that I maybe want to use**
When I run thesis_workflow.py I get also a lot of plots generated in the combined folder. From this I only want to use the distribution overlay dot png and I want to have the distribution overlays as separate graphics, not as one combined graphic. I like the distribution overlay graphics but I don't want to have them separate.

In the dashboards, the paradigm comparison seems a bit weird to me. Maybe I need this in another way represented, as well as I need way more explanation to this regarding the formulas that I used and how this is calculated so that I have a clear description of what's going on there. Also including the formulas that I used inside the code.  