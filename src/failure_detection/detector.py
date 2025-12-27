   
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np
from datetime import datetime

from .taxonomy import FailureType, FailureReport

class FailureDetector(ABC):
           
    failure_type: FailureType = None
    
    def __init__(self, window_size: int = 100):
                   
        self.window_size = window_size
        self._step = 0
        self._episode = 0
        self._active = True
    
    @abstractmethod
    def update(self, metrics: Dict[str, Any]) -> None:
                   
        self._step = metrics.get("step", self._step + 1)
        self._episode = metrics.get("episode", self._episode)
    
    @abstractmethod
    def detect(self) -> Optional[FailureReport]:
                   
        pass
    
    def reset(self) -> None:
                                   
        self._step = 0
        self._episode = 0
    
    def set_active(self, active: bool) -> None:
                                             
        self._active = active
    
    def _create_report(
        self,
        severity: str,
        signal_values: Dict[str, float],
        description: str,
        recommended_fix: str,
        **additional_info
    ) -> FailureReport:
                                                
        return FailureReport(
            failure_type=self.failure_type,
            severity=severity,
            timestamp=datetime.now(),
            step=self._step,
            episode=self._episode,
            signal_values=signal_values,
            description=description,
            recommended_fix=recommended_fix,
            additional_info=additional_info
        )

class CombinedFailureDetector:
           
    def __init__(self, detectors: List[FailureDetector]):
                   
        self.detectors = detectors
        self._all_reports: List[FailureReport] = []
    
    def update(self, metrics: Dict[str, Any]) -> None:
                                                    
        for detector in self.detectors:
            if detector._active:
                detector.update(metrics)
    
    def detect_all(self) -> List[FailureReport]:
                   
        reports = []
        for detector in self.detectors:
            if detector._active:
                report = detector.detect()
                if report is not None:
                    reports.append(report)
                    self._all_reports.append(report)
        return reports
    
    def get_all_reports(self) -> List[FailureReport]:
                                                 
        return self._all_reports.copy()
    
    def get_summary(self) -> Dict[str, int]:
                                                      
        counts = {}
        for report in self._all_reports:
            key = report.failure_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def reset(self) -> None:
                                  
        for detector in self.detectors:
            detector.reset()
        self._all_reports.clear()
    
    def add_detector(self, detector: FailureDetector) -> None:
                                 
        self.detectors.append(detector)
    
    def remove_detector(self, failure_type: FailureType) -> None:
                                                  
        self.detectors = [d for d in self.detectors if d.failure_type != failure_type]
