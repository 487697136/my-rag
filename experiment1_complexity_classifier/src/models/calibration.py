"""
置信度校准方法实现

包含以下校准技术:
1. Temperature Scaling - 核心创新方法
2. Top-Versus-All + Platt Scaling
3. Top-Versus-All + Isotonic Regression  
4. TvA + Temperature Scaling组合方法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling(BaseEstimator):
    """温度缩放校准方法 - 核心创新
    
    通过学习一个全局温度参数T来校准模型输出概率：
    p_calibrated = softmax(logits / T)
    
    这是本实验的核心创新点，预期能显著改善ECE指标。
    """
    
    def __init__(
        self,
        temperature_range: Tuple[float, float] = (0.1, 5.0),
        num_search_points: int = 50,
        optimization_method: str = "grid_search",  # grid_search, minimize
        objective: str = "ECE",  # ECE, NLL, Brier
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        cross_validation: bool = True,
        n_folds: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            temperature_range: 温度参数搜索范围
            num_search_points: 网格搜索点数
            optimization_method: 优化方法
            objective: 优化目标函数
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            cross_validation: 是否使用交叉验证
            n_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.temperature_range = temperature_range
        self.num_search_points = num_search_points
        self.optimization_method = optimization_method
        self.objective = objective
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cross_validation = cross_validation
        self.n_folds = n_folds
        self.random_state = random_state
        
        # 学习到的参数
        self.temperature_ = 1.0
        self.optimization_history_ = []
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        """拟合温度参数
        
        Args:
            logits: 模型原始logits [n_samples, n_classes]
            y_true: 真实标签 [n_samples]
        """
        logger.info(f"开始温度缩放校准拟合，样本数量: {len(logits)}")
        
        if self.optimization_method == "grid_search":
            self.temperature_ = self._grid_search_temperature(logits, y_true)
        elif self.optimization_method == "minimize":
            self.temperature_ = self._optimize_temperature(logits, y_true)
        else:
            raise ValueError(f"未知的优化方法: {self.optimization_method}")
        
        self.is_fitted_ = True
        logger.info(f"温度缩放校准完成，最优温度: {self.temperature_:.4f}")
        return self
    
    def _grid_search_temperature(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """网格搜索最优温度"""
        temperatures = np.linspace(
            self.temperature_range[0],
            self.temperature_range[1],
            self.num_search_points
        )
        
        best_temp = 1.0
        best_score = float('inf')
        
        for temp in temperatures:
            score = self._evaluate_temperature(temp, logits, y_true)
            self.optimization_history_.append({'temperature': temp, 'score': score})
            
            if score < best_score:
                best_score = score
                best_temp = temp
        
        return best_temp
    
    def _optimize_temperature(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """使用scipy.optimize优化温度"""
        
        def objective_func(temp):
            return self._evaluate_temperature(temp[0], logits, y_true)
        
        result = minimize(
            objective_func,
            x0=[1.0],
            bounds=[self.temperature_range],
            method='L-BFGS-B',
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        return result.x[0]
    
    def _evaluate_temperature(self, temperature: float, logits: np.ndarray, y_true: np.ndarray) -> float:
        """评估给定温度的校准效果"""
        calibrated_probs = self._apply_temperature(logits, temperature)
        
        if self.objective == "ECE":
            return self._calculate_ece(calibrated_probs, y_true)
        elif self.objective == "NLL":
            return self._calculate_nll(calibrated_probs, y_true)
        elif self.objective == "Brier":
            return self._calculate_brier(calibrated_probs, y_true)
        else:
            raise ValueError(f"未知的目标函数: {self.objective}")
    
    def _apply_temperature(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """应用温度缩放"""
        scaled_logits = logits / temperature
        return F.softmax(torch.tensor(scaled_logits), dim=1).numpy()
    
    def _calculate_ece(self, probs: np.ndarray, y_true: np.ndarray, num_bins: int = 15) -> float:
        """计算Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true)
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_nll(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """计算Negative Log Likelihood"""
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        return -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))
    
    def _calculate_brier(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """计算Brier Score"""
        one_hot = np.eye(probs.shape[1])[y_true]
        return np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """应用校准变换"""
        if not self.is_fitted_:
            raise ValueError("校准器尚未拟合，请先调用fit方法")
        
        return self._apply_temperature(logits, self.temperature_)
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """预测校准后的概率"""
        return self.transform(logits)


class TopVersusAllPlatt(BaseEstimator):
    """Top-Versus-All + Platt Scaling校准方法
    
    将多分类问题转化为二分类（最大logit对应的类别是否为真实类别），
    然后使用Platt scaling进行校准。
    """
    
    def __init__(
        self,
        regularization: float = 1e-6,
        max_iterations: int = 1000,
        solver: str = "lbfgs",
        C_values: Optional[List[float]] = None,
        cross_validation: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            regularization: 正则化参数
            max_iterations: 最大迭代次数
            solver: 求解器类型
            C_values: 交叉验证的C值列表
            cross_validation: 交叉验证折数
            random_state: 随机种子
        """
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.solver = solver
        self.C_values = C_values or [0.001, 0.01, 0.1, 1.0, 10.0]
        self.cross_validation = cross_validation
        self.random_state = random_state
        
        # 拟合的模型
        self.platt_regressor_ = None
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'TopVersusAllPlatt':
        """拟合Platt scaling参数"""
        logger.info("开始Top-vs-All + Platt Scaling校准拟合")
        
        # 创建二分类代理任务
        max_logits = np.max(logits, axis=1)
        predicted_classes = np.argmax(logits, axis=1)
        binary_labels = (predicted_classes == y_true).astype(int)
        
        # 寻找最佳C值
        best_c = self._find_best_c(max_logits.reshape(-1, 1), binary_labels)
        
        # 训练最终模型
        self.platt_regressor_ = LogisticRegression(
            C=best_c,
            solver=self.solver,
            max_iter=self.max_iterations,
            random_state=self.random_state
        )
        
        self.platt_regressor_.fit(max_logits.reshape(-1, 1), binary_labels)
        self.is_fitted_ = True
        
        logger.info(f"Platt scaling校准完成，最佳C值: {best_c}")
        return self
    
    def _find_best_c(self, X: np.ndarray, y: np.ndarray) -> float:
        """通过交叉验证寻找最佳C值"""
        from sklearn.model_selection import cross_val_score
        
        best_c = 1.0
        best_score = -float('inf')
        
        for c in self.C_values:
            clf = LogisticRegression(
                C=c,
                solver=self.solver,
                max_iter=self.max_iterations,
                random_state=self.random_state
            )
            
            scores = cross_val_score(clf, X, y, cv=self.cross_validation, scoring='accuracy')
            avg_score = scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                best_c = c
        
        return best_c
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """应用校准变换"""
        if not self.is_fitted_:
            raise ValueError("校准器尚未拟合，请先调用fit方法")
        
        max_logits = np.max(logits, axis=1)
        predicted_classes = np.argmax(logits, axis=1)
        
        # 获取校准后的置信度
        calibrated_confidence = self.platt_regressor_.predict_proba(max_logits.reshape(-1, 1))[:, 1]
        
        # 重构概率分布
        calibrated_probs = np.zeros_like(logits, dtype=float)
        n_classes = logits.shape[1]
        
        for i in range(len(logits)):
            pred_class = predicted_classes[i]
            confidence = calibrated_confidence[i]
            
            # 将校准后的置信度分配给预测类别
            calibrated_probs[i, pred_class] = confidence
            # 剩余概率平均分配给其他类别
            remaining_prob = (1 - confidence) / (n_classes - 1)
            for j in range(n_classes):
                if j != pred_class:
                    calibrated_probs[i, j] = remaining_prob
        
        return calibrated_probs
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """预测校准后的概率"""
        return self.transform(logits)


class TopVersusAllIsotonic(BaseEstimator):
    """Top-Versus-All + Isotonic Regression校准方法
    
    类似于Platt scaling，但使用非参数的等距回归进行校准。
    """
    
    def __init__(
        self,
        increasing: bool = True,
        out_of_bounds: str = "clip",
        smoothing: bool = False,
        smoothing_factor: float = 0.1
    ):
        """
        Args:
            increasing: 是否强制单调递增
            out_of_bounds: 边界外处理方式
            smoothing: 是否进行平滑
            smoothing_factor: 平滑因子
        """
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds
        self.smoothing = smoothing
        self.smoothing_factor = smoothing_factor
        
        # 拟合的模型
        self.isotonic_regressor_ = None
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'TopVersusAllIsotonic':
        """拟合Isotonic regression参数"""
        logger.info("开始Top-vs-All + Isotonic Regression校准拟合")
        
        # 创建二分类代理任务
        max_logits = np.max(logits, axis=1)
        predicted_classes = np.argmax(logits, axis=1)
        binary_labels = (predicted_classes == y_true).astype(int)
        
        # 训练Isotonic regression
        self.isotonic_regressor_ = IsotonicRegression(
            increasing=self.increasing,
            out_of_bounds=self.out_of_bounds
        )
        
        self.isotonic_regressor_.fit(max_logits, binary_labels)
        self.is_fitted_ = True
        
        logger.info("Isotonic regression校准完成")
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """应用校准变换"""
        if not self.is_fitted_:
            raise ValueError("校准器尚未拟合，请先调用fit方法")
        
        max_logits = np.max(logits, axis=1)
        predicted_classes = np.argmax(logits, axis=1)
        
        # 获取校准后的置信度
        calibrated_confidence = self.isotonic_regressor_.predict(max_logits)
        
        # 重构概率分布
        calibrated_probs = np.zeros_like(logits, dtype=float)
        n_classes = logits.shape[1]
        
        for i in range(len(logits)):
            pred_class = predicted_classes[i]
            confidence = calibrated_confidence[i]
            
            # 确保置信度在[0,1]范围内
            confidence = np.clip(confidence, 0, 1)
            
            calibrated_probs[i, pred_class] = confidence
            remaining_prob = (1 - confidence) / (n_classes - 1)
            for j in range(n_classes):
                if j != pred_class:
                    calibrated_probs[i, j] = remaining_prob
        
        return calibrated_probs
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """预测校准后的概率"""
        return self.transform(logits)


class TvATemperatureScaling(BaseEstimator):
    """TvA + Temperature Scaling组合校准方法
    
    先应用Top-vs-All校准，再应用温度缩放。
    """
    
    def __init__(
        self,
        tva_method: str = "platt",  # platt, isotonic
        temperature_range: Tuple[float, float] = (0.5, 3.0),
        num_temp_points: int = 30,
        combination: str = "sequential"  # sequential, ensemble
    ):
        """
        Args:
            tva_method: TvA校准方法
            temperature_range: 温度搜索范围
            num_temp_points: 温度搜索点数
            combination: 组合方式
        """
        self.tva_method = tva_method
        self.temperature_range = temperature_range
        self.num_temp_points = num_temp_points
        self.combination = combination
        
        # 校准器组件
        self.tva_calibrator_ = None
        self.temperature_calibrator_ = None
        self.is_fitted_ = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'TvATemperatureScaling':
        """拟合组合校准器"""
        logger.info(f"开始TvA + Temperature组合校准拟合，TvA方法: {self.tva_method}")
        
        # 第一步：TvA校准
        if self.tva_method == "platt":
            self.tva_calibrator_ = TopVersusAllPlatt()
        elif self.tva_method == "isotonic":
            self.tva_calibrator_ = TopVersusAllIsotonic()
        else:
            raise ValueError(f"未知的TvA方法: {self.tva_method}")
        
        self.tva_calibrator_.fit(logits, y_true)
        
        # 第二步：温度缩放
        tva_probs = self.tva_calibrator_.transform(logits)
        # 将概率转回logits进行温度缩放
        tva_logits = np.log(np.clip(tva_probs, 1e-8, 1 - 1e-8))
        
        self.temperature_calibrator_ = TemperatureScaling(
            temperature_range=self.temperature_range,
            num_search_points=self.num_temp_points
        )
        self.temperature_calibrator_.fit(tva_logits, y_true)
        
        self.is_fitted_ = True
        logger.info("TvA + Temperature组合校准完成")
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """应用组合校准变换"""
        if not self.is_fitted_:
            raise ValueError("校准器尚未拟合，请先调用fit方法")
        
        # 第一步：TvA校准
        tva_probs = self.tva_calibrator_.transform(logits)
        
        # 第二步：温度缩放
        tva_logits = np.log(np.clip(tva_probs, 1e-8, 1 - 1e-8))
        final_probs = self.temperature_calibrator_.transform(tva_logits)
        
        return final_probs
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """预测校准后的概率"""
        return self.transform(logits)


class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    """校准分类器包装类
    
    将任意分类器与校准方法组合，提供统一接口。
    """
    
    def __init__(
        self,
        base_classifier,
        calibration_method: str = "temperature_scaling",
        calibration_params: Optional[Dict] = None
    ):
        """
        Args:
            base_classifier: 基础分类器
            calibration_method: 校准方法名称
            calibration_params: 校准参数
        """
        self.base_classifier = base_classifier
        self.calibration_method = calibration_method
        self.calibration_params = calibration_params or {}
        
        # 校准器
        self.calibrator_ = None
        self.is_fitted_ = False
        self.classes_ = None
    
    def fit(self, X, y, X_cal=None, y_cal=None):
        """训练分类器和校准器
        
        Args:
            X: 训练数据
            y: 训练标签
            X_cal: 校准数据（可选）
            y_cal: 校准标签（可选）
        """
        # 训练基础分类器
        self.base_classifier.fit(X, y)
        self.classes_ = self.base_classifier.classes_
        
        # 准备校准数据
        if X_cal is None or y_cal is None:
            # 使用训练数据进行校准（可能会过拟合）
            X_cal, y_cal = X, y
            logger.warning("使用训练数据进行校准，可能存在过拟合风险")
        
        # 获取校准数据的logits
        calibration_logits = self.base_classifier.get_logits(X_cal)
        
        # 初始化校准器
        self.calibrator_ = self._create_calibrator()
        
        # 拟合校准器
        self.calibrator_.fit(calibration_logits, y_cal)
        
        self.is_fitted_ = True
        return self
    
    def _create_calibrator(self):
        """创建校准器实例"""
        if self.calibration_method == "temperature_scaling":
            return TemperatureScaling(**self.calibration_params)
        elif self.calibration_method == "tva_platt":
            return TopVersusAllPlatt(**self.calibration_params)
        elif self.calibration_method == "tva_isotonic":
            return TopVersusAllIsotonic(**self.calibration_params)
        elif self.calibration_method == "tva_temperature":
            return TvATemperatureScaling(**self.calibration_params)
        else:
            raise ValueError(f"未知的校准方法: {self.calibration_method}")
    
    def predict(self, X):
        """预测类别"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        """预测校准后的概率"""
        if not self.is_fitted_:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取原始logits
        logits = self.base_classifier.get_logits(X)
        
        # 应用校准
        calibrated_probs = self.calibrator_.transform(logits)
        
        return calibrated_probs
    
    def get_logits(self, X):
        """获取原始logits"""
        return self.base_classifier.get_logits(X)
    
    def get_calibrated_logits(self, X):
        """获取校准后的logits"""
        calibrated_probs = self.predict_proba(X)
        calibrated_probs = np.clip(calibrated_probs, 1e-8, 1 - 1e-8)
        return np.log(calibrated_probs)