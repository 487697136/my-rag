# 校准方法对比报告

生成时间: 2025-08-10 17:12:15

## 元信息

- 随机种子: 42
- ECE分箱: 15
- 分箱策略: equal_width
- 校准来源: validation_set

## 校准效果对比

Method,ECE,MCE,Brier Score,NLL,Accuracy,Mean Confidence,ECE_Improvement
tva_isotonic,0.0402,0.5324,0.2432,0.5013,0.8368,0.8705,60.63
temperature_scaling,0.0447,0.4562,0.2144,0.3612,0.8345,0.8738,56.22
tva_platt,0.0606,0.5233,0.2454,0.4491,0.8345,0.8677,40.65
tva_temperature,0.0607,0.5259,0.2448,0.4486,0.8345,0.8641,40.55
uncalibrated,0.1021,0.3711,0.2447,0.4719,0.8345,0.9366,0.0


## 关键发现\n\n- **最佳校准方法**: tva_isotonic (ECE = 0.0402)\n- **最大ECE改进**: tva_isotonic (60.6%)\n- **达到ECE目标(≤0.08)的方法**: 4个\n  - tva_isotonic: 0.0402\n  - temperature_scaling: 0.0447\n  - tva_platt: 0.0606\n  - tva_temperature: 0.0607\n\n## 方法分析\n\n### tva_isotonic\n- ECE: 0.0402\n- MCE: 0.5324\n- Brier Score: 0.2432\n- 准确率: 0.8368\n- ECE改进: 60.6%\n- 评价: 优秀的校准质量\n\n### temperature_scaling\n- ECE: 0.0447\n- MCE: 0.4562\n- Brier Score: 0.2144\n- 准确率: 0.8345\n- ECE改进: 56.2%\n- 评价: 优秀的校准质量\n\n### tva_platt\n- ECE: 0.0606\n- MCE: 0.5233\n- Brier Score: 0.2454\n- 准确率: 0.8345\n- ECE改进: 40.6%\n- 评价: 良好的校准质量\n\n### tva_temperature\n- ECE: 0.0607\n- MCE: 0.5259\n- Brier Score: 0.2448\n- 准确率: 0.8345\n- ECE改进: 40.5%\n- 评价: 良好的校准质量\n\n### uncalibrated\n- ECE: 0.1021\n- MCE: 0.3711\n- Brier Score: 0.2447\n- 准确率: 0.8345\n- 评价: 中等的校准质量\n\n