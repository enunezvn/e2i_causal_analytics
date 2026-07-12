/**
 * Admin React Query Hooks (/api/admin/*)
 * =======================================
 *
 * Mutations invalidate the admin query group — no optimistic updates (spec:
 * refetch after mutation).
 *
 * @module hooks/api/use-admin
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  deleteUser,
  disableUser,
  enableUser,
  getAuditFeed,
  getLlmUsage,
  getPlatformActivity,
  getUserActivity,
  inviteUser,
  listUsers,
  recoveryLink,
  reinviteUser,
  updateUser,
} from '@/api/admin';
import type { InviteRequest, UpdateUserRequest } from '@/types/admin';

export function useAdminUsers() {
  return useQuery({
    queryKey: queryKeys.admin.users(),
    queryFn: listUsers,
  });
}

export function useUserActivity(userId: string | null, days = 90) {
  return useQuery({
    queryKey: queryKeys.admin.userActivity(userId ?? 'none', days),
    queryFn: () => getUserActivity(userId as string, days),
    enabled: Boolean(userId),
  });
}

export function usePlatformActivity(days = 30) {
  return useQuery({
    queryKey: queryKeys.admin.platformActivity(days),
    queryFn: () => getPlatformActivity(days),
  });
}

export function useAuditFeed(days = 30) {
  return useQuery({
    queryKey: queryKeys.admin.auditFeed(days),
    queryFn: () => getAuditFeed(days),
  });
}

export function useLlmUsage(days = 30) {
  return useQuery({
    queryKey: queryKeys.admin.llmUsage(days),
    queryFn: () => getLlmUsage(days),
  });
}

function useInvalidateAdmin() {
  const queryClient = useQueryClient();
  return () => queryClient.invalidateQueries({ queryKey: queryKeys.admin.all() });
}

export function useInviteUser() {
  const invalidate = useInvalidateAdmin();
  return useMutation({
    mutationFn: (body: InviteRequest) => inviteUser(body),
    onSuccess: invalidate,
  });
}

export function useReinviteUser() {
  return useMutation({ mutationFn: (userId: string) => reinviteUser(userId) });
}

export function useRecoveryLink() {
  return useMutation({ mutationFn: (userId: string) => recoveryLink(userId) });
}

export function useUpdateUser() {
  const invalidate = useInvalidateAdmin();
  return useMutation({
    mutationFn: ({ userId, body }: { userId: string; body: UpdateUserRequest }) =>
      updateUser(userId, body),
    onSuccess: invalidate,
  });
}

export function useDisableUser() {
  const invalidate = useInvalidateAdmin();
  return useMutation({ mutationFn: disableUser, onSuccess: invalidate });
}

export function useEnableUser() {
  const invalidate = useInvalidateAdmin();
  return useMutation({ mutationFn: enableUser, onSuccess: invalidate });
}

export function useDeleteUser() {
  const invalidate = useInvalidateAdmin();
  return useMutation({ mutationFn: deleteUser, onSuccess: invalidate });
}
