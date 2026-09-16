export interface DecisionRow {
  id: number;
  transaction_id: string;
  model_score: number;
  decision: "approve" | "review" | "block";
  triggered_rules: string[];
  decision_source: "model" | "rule" | "rule_override";
  model_run_id: string;
  shap_top_features: Record<string, number>;
  created_at: string;
}

export interface DecisionsListResponse {
  items: DecisionRow[];
  next_cursor: number | null;
}

export interface DecisionsStatsBucket {
  minute: string;
  decision: "approve" | "review" | "block";
  count: number;
}

export interface DecisionsStatsResponse {
  total: number;
  approve_count: number;
  review_count: number;
  block_count: number;
  avg_score: number;
  buckets: DecisionsStatsBucket[];
}

export interface CostCurvePoint {
  t_review: number;
  cost: number;
}

export interface ModelMetadataResponse {
  model_run_id: string;
  deployed_model_name: string;
  val_pr_auc: number;
  test_pr_auc: number;
  calibration_method: string;
  t_review: number;
  t_block: number;
  cost_curve: CostCurvePoint[];
}

export interface ModelComparisonCandidate {
  name: string;
  val_pr_auc: number;
  is_deployed: boolean;
}

export interface CalibrationPoint {
  mean_predicted: number;
  fraction_positive: number;
}

export interface ShapImportance {
  feature: string;
  mean_abs_shap: number;
}

export interface ModelComparisonResponse {
  candidates: ModelComparisonCandidate[];
  calibration_curve: CalibrationPoint[];
  shap_importances: ShapImportance[];
}

export interface InvestigationResponse {
  transaction_id: string;
  model_score: number;
  decision: "approve" | "review" | "block";
  triggered_rules: string[];
  decision_source: "model" | "rule" | "rule_override";
  model_run_id: string;
  shap_top_features: Record<string, number>;
  created_at: string;
}
