/**
 * Admin API Client (/api/admin/*)
 * ===============================
 *
 * TypeScript API client functions for the admin user-management endpoints.
 * All endpoints require an admin JWT — the shared apiClient attaches it.
 *
 * @module api/admin
 */

import { get, post, patch, del } from '@/lib/api-client';
import type {
  AdminUsersResponse,
  AuditFeedResponse,
  InviteRequest,
  LinkResponse,
  LlmUsageResponse,
  PlatformActivityResponse,
  UpdateUserRequest,
  UserActivityResponse,
} from '@/types/admin';

const BASE = '/admin';

export function listUsers(): Promise<AdminUsersResponse> {
  return get<AdminUsersResponse>(`${BASE}/users`);
}

export function inviteUser(body: InviteRequest): Promise<LinkResponse> {
  return post<LinkResponse, InviteRequest>(`${BASE}/users/invite`, body);
}

export function reinviteUser(userId: string): Promise<LinkResponse> {
  return post<LinkResponse>(`${BASE}/users/${userId}/reinvite`);
}

export function recoveryLink(userId: string): Promise<LinkResponse> {
  return post<LinkResponse>(`${BASE}/users/${userId}/recovery-link`);
}

export function updateUser(
  userId: string,
  body: UpdateUserRequest
): Promise<{ user_id: string; role: string; brands: string[] }> {
  return patch<{ user_id: string; role: string; brands: string[] }, UpdateUserRequest>(
    `${BASE}/users/${userId}`,
    body
  );
}

export function disableUser(userId: string): Promise<{ user_id: string; status: string }> {
  return post<{ user_id: string; status: string }>(`${BASE}/users/${userId}/disable`);
}

export function enableUser(userId: string): Promise<{ user_id: string; status: string }> {
  return post<{ user_id: string; status: string }>(`${BASE}/users/${userId}/enable`);
}

export function deleteUser(
  userId: string
): Promise<{ user_id: string; email: string; deleted: boolean }> {
  return del<{ user_id: string; email: string; deleted: boolean }>(`${BASE}/users/${userId}`);
}

export function getUserActivity(userId: string, days = 90): Promise<UserActivityResponse> {
  return get<UserActivityResponse>(`${BASE}/users/${userId}/activity`, {
    params: { days },
  });
}

export function getPlatformActivity(days = 30): Promise<PlatformActivityResponse> {
  return get<PlatformActivityResponse>(`${BASE}/activity/overview`, { params: { days } });
}

export function getAuditFeed(days = 30): Promise<AuditFeedResponse> {
  return get<AuditFeedResponse>(`${BASE}/audit`, { params: { days } });
}

export function getLlmUsage(days = 30): Promise<LlmUsageResponse> {
  return get<LlmUsageResponse>(`${BASE}/observability/llm-usage`, { params: { days } });
}
