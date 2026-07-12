/** Admin user management types (mirrors /api/admin/* responses). */

export type AdminRole = 'viewer' | 'analyst' | 'operator' | 'admin';
export type AdminUserStatus = 'active' | 'invited' | 'disabled';
export type AdminBrand = 'Kisqali' | 'Fabhalta' | 'Remibrutinib' | 'all';

export const ADMIN_ROLES: AdminRole[] = ['viewer', 'analyst', 'operator', 'admin'];
export const ADMIN_BRANDS: AdminBrand[] = ['Kisqali', 'Fabhalta', 'Remibrutinib', 'all'];

// Mirrors the backend RBAC hierarchy in src/api/dependencies/auth.py
// (ADMIN > OPERATOR > ANALYST > VIEWER; higher roles inherit lower permissions).
export const ROLE_DESCRIPTIONS: Record<AdminRole, string> = {
  viewer: 'Read-only access to dashboards, KPIs, and reports.',
  analyst: 'Can also run analyses: causal, gap, and segment.',
  operator: 'Can also manage experiments, feedback learning, and the digital twin.',
  admin: 'Full system management: user administration, cache, and model retraining.',
};

export interface AdminUser {
  id: string;
  email: string;
  full_name: string | null;
  role: AdminRole;
  brands: string[];
  status: AdminUserStatus;
  created_at: string | null;
  last_sign_in_at: string | null;
  total_conversations: number;
  total_messages: number;
  last_active_at: string | null;
}

export interface AdminUsersResponse {
  users: AdminUser[];
}

export interface InviteRequest {
  email: string;
  role: AdminRole;
  brands: string[];
  full_name?: string;
}

export interface LinkResponse {
  user_id: string;
  email: string;
  invite_link: string;
  link_type: 'invite' | 'recovery';
}

export interface UpdateUserRequest {
  role?: AdminRole;
  brands?: string[];
  full_name?: string;
}

export interface AuthEventBucket {
  day: string;
  event_type: string;
  event_count: number;
}

export interface ApiActivityRow {
  endpoint_group: string;
  http_method: string;
  bucket_minute: string;
  request_count: number;
}

export interface UserActivityResponse {
  user_id: string;
  email: string;
  auth_events: AuthEventBucket[];
  api_activity: ApiActivityRow[];
  recent_events: { occurred_at: string; action: string }[];
  chat: {
    total_conversations: number;
    total_messages: number;
    last_active_at: string | null;
  };
}

export interface PlatformActivityResponse {
  days: { day: string; logins: number; active_users: number }[];
}

export interface AuditFeedResponse {
  events: {
    event_id: string;
    event_type: string;
    severity: string;
    timestamp: string;
    message: string;
    user_email: string | null;
    resource_id: string | null;
    metadata: Record<string, unknown>;
  }[];
}

// --- LLM observability (mirrors GET /api/admin/observability/llm-usage) ---

export interface LlmUsageSummary {
  total_cost_usd: number | null;
  input_tokens: number;
  output_tokens: number;
  calls: number;
  distinct_users: number;
  days: number;
  tracking_since: string | null;
}

export interface LlmDailyUsage {
  date: string;
  chat_cost_usd: number;
  platform_cost_usd: number;
  tokens: number;
}

export interface LlmUserUsage {
  user_id: string;
  email: string | null;
  sessions: number;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  models: string[];
}

export interface LlmSessionUsage {
  session_id: string;
  title: string | null;
  started_at: string | null;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  models: string[];
}

export interface LlmPlatformUsage {
  surface: string;
  component: string | null;
  model: string;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
}

export interface LlmUsageResponse {
  summary: LlmUsageSummary;
  daily: LlmDailyUsage[];
  by_user: LlmUserUsage[];
  sessions: Record<string, LlmSessionUsage[]>;
  platform: LlmPlatformUsage[];
  pricing_version: string;
  unpriced_models: string[];
}
