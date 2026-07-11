/** Admin user management types (mirrors /api/admin/* responses). */

export type AdminRole = 'viewer' | 'analyst' | 'operator' | 'admin';
export type AdminUserStatus = 'active' | 'invited' | 'disabled';
export type AdminBrand = 'Kisqali' | 'Fabhalta' | 'Remibrutinib' | 'all';

export const ADMIN_ROLES: AdminRole[] = ['viewer', 'analyst', 'operator', 'admin'];
export const ADMIN_BRANDS: AdminBrand[] = ['Kisqali', 'Fabhalta', 'Remibrutinib', 'all'];

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
