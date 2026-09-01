CREDIT CARD DEFAULT PREDICTION AND BEHAVIOUR SCORING USING MACHINE LEARNING

Overview: This project focuses on developing a high-performance machine learning system to identify high-risk customers by analyzing financial, transactional, and credit-bureau data. It includes thorough data exploration, advanced missing-value imputation, engineered feature creation, imbalance handling, and the evaluation of multiple models—ultimately delivering an optimized ensemble classifier capable of reliably detecting potential defaults or non-compliance cases.

Key Steps:

1. Data Understanding & EDA
   - Explored 140K+ records across on-us, transaction, and bureau attributes.
   - Identified missing values, outliers, and key correlated behavioral features.
   - Missing Value Imputation
   - Compared MICE and MCMC.
   - MCMC chosen for better multivariate consistency.
2. Feature Engineering
   - AutoFE-based transformations, PCA, and interaction features.
   - Feature ranking via ANOVA F-test + RFE.
   - Final: 282 optimized features.
3. Class Imbalance Handling
   - Tested SMOTE, ADASYN, Tomek Links, NearMiss, and no sampling.
   - Best approach: No sampling + Logit Shift, threshold tuned using Youden’s J.
4. Modeling
   - Models trained: XGBoost, CatBoost, LightGBM, Logistic Regression, Random Forest, SVM, Decision Trees.
   - Best model: Voting Classifier
5. Performance: 
   a.) ROC-AUC: 0.7832 
   b.) Recall: 0.23
