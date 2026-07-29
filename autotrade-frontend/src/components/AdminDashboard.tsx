import { useState, useEffect, useCallback } from 'react';
import { ShieldCheck, Users, Trash2, Upload, Loader2, Search, CheckCircle, AlertTriangle, X, FileText } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../lib/auth';
import { API_URL } from '../lib/api';

interface UserProfile {
  id: string;
  auth_user_id: string | null;
  full_name: string | null;
  email: string | null;
  role: string;
  created_at: string;
}

interface RegistryDocument {
  id: string;
  document_name: string;
  document_type: string;
  created_at: string;
  storage_url: string;
  status?: string;
}

// ─────────────────────────────────────────────
// Upload Stage Types
// ─────────────────────────────────────────────
export type UploadStatus = 'idle' | 'uploading' | 'registering' | 'vectorizing' | 'success' | 'error';

export interface UploadData {
  filename: string;
  message?: string;
}

type UploadStage = 'uploading' | 'registering' | 'vectorizing' | null;

interface UploadResult {
  type: 'success' | 'error';
  filename: string;
  message?: string;
}

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState<'users' | 'registry'>('users');

  return (
    <div className="h-full p-8 overflow-y-auto">
      {/* Header */}
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center space-x-3 mb-2">
          <div className="p-2 bg-amber-100 rounded-xl">
            <ShieldCheck size={22} className="text-amber-600" />
          </div>
          <div>
            <h2 className="text-2xl font-serif text-stone-800">Admin Dashboard</h2>
            <p className="text-stone-500 text-sm mt-0.5">Manage users and uploaded documents</p>
          </div>
        </div>

        {/* Tab Bar */}
        <div className="flex space-x-1 mt-8 bg-stone-100 rounded-xl p-1 w-fit">
          <button
            onClick={() => setActiveTab('users')}
            className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${activeTab === 'users'
              ? 'bg-white text-stone-900 shadow-sm'
              : 'text-stone-500 hover:text-stone-700'
              }`}
          >
            <Users size={16} />
            <span>Users</span>
          </button>
          <button
            onClick={() => setActiveTab('registry')}
            className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${activeTab === 'registry'
              ? 'bg-white text-stone-900 shadow-sm'
              : 'text-stone-500 hover:text-stone-700'
              }`}
          >
            <FileText size={16} />
            <span>File Manager</span>
          </button>
        </div>

        {/* Tab Content */}
        <div className="mt-6">
          {activeTab === 'users' && <UsersTab />}
          {activeTab === 'registry' && <FileManager />}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Users Tab
// ─────────────────────────────────────────────
function UsersTab() {
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [userToProcess, setUserToProcess] = useState<UserProfile | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('user_profiles')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      setUsers(data || []);
    } catch (err) {
      console.error('Failed to fetch users:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRoleChange = async (targetUserId: string, newRole: string) => {
    if (!currentUser || targetUserId === currentUser.id) return;
    setProcessingId(targetUserId);
    try {
      const response = await fetch(`${API_URL}/admin/users/${targetUserId}/role`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole, admin_id: currentUser.id }),
      });
      const result = await response.json();
      if (result.status === 'success') {
        setUsers(prev => prev.map(u => u.id === targetUserId ? { ...u, role: newRole } : u));
      }
    } catch (err) {
      console.error('Role change failed:', err);
    } finally {
      setProcessingId(null);
    }
  };

  const handleDeleteUser = (targetUser: UserProfile) => {
    if (!currentUser || targetUser.auth_user_id === currentUser.id) return;
    setUserToProcess(targetUser);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!currentUser || !userToProcess) return;
    const targetUser = userToProcess;
    setIsDeleteModalOpen(false);
    setUserToProcess(null);
    setProcessingId(targetUser.id);
    try {
      const response = await fetch(`${API_URL}/admin/users/${targetUser.id}?admin_id=${currentUser.id}`, {
        method: 'DELETE',
      });
      const result = await response.json();
      if (response.ok || result.status === 'success') {
        setUsers(prev => prev.filter(u => u.id !== targetUser.id));
      }
    } catch (err) {
      console.error('Delete failed:', err);
    } finally {
      setProcessingId(null);
    }
  };

  const filteredUsers = users.filter(u =>
    (u.full_name || '').toLowerCase().includes(search.toLowerCase()) ||
    (u.email || '').toLowerCase().includes(search.toLowerCase())
  );

  if (loading) {
    return <LoadingSkeleton rows={5} />;
  }

  return (
    <div className="bg-white rounded-3xl border border-stone-200 shadow-sm overflow-hidden">
      {/* Search Bar */}
      <div className="p-5 border-b border-stone-100 flex items-center justify-between">
        <div className="relative flex-1 max-w-sm">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search users..."
            className="w-full bg-stone-50 border border-stone-200 rounded-xl pl-9 pr-4 py-2 text-sm text-stone-800 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 transition-all placeholder:text-stone-400"
          />
        </div>
        <div className="text-sm text-stone-500 ml-4">
          {filteredUsers.length} user{filteredUsers.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-stone-100">
              <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">User</th>
              <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Email</th>
              <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Role</th>
              <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Joined</th>
              <th className="text-right px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-50">
            {filteredUsers.map((user) => (
              <tr key={user.id} className="hover:bg-stone-50/50 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-9 h-9 rounded-full bg-stone-200 flex items-center justify-center text-stone-600 uppercase font-bold text-xs flex-shrink-0">
                      {(user.full_name?.[0] || user.email?.[0] || 'U').toUpperCase()}
                    </div>
                    <span className="font-medium text-stone-800 text-sm">
                      {user.full_name || 'Unnamed'}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 text-sm text-stone-600">
                  {user.email || '\u2014'}
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-2">
                    <select
                      value={user.role || 'user'}
                      onChange={(e) => handleRoleChange(user.id, e.target.value)}
                      disabled={user.auth_user_id === currentUser?.id || processingId === user.id}
                      className={`text-xs font-semibold rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-amber-500/20 disabled:opacity-50 transition-colors cursor-pointer ${
                        user.role === 'admin'
                          ? 'bg-amber-100 text-amber-700 border border-amber-200 hover:bg-amber-200'
                          : 'bg-stone-100 text-stone-600 border border-stone-200 hover:bg-stone-200'
                      }`}
                    >
                      <option value="user">User</option>
                      <option value="admin">Admin</option>
                    </select>
                    {processingId === user.id && <Loader2 size={14} className="animate-spin text-stone-400" />}
                  </div>
                </td>
                <td className="px-6 py-4 text-sm text-stone-500">
                  {user.created_at
                    ? new Date(user.created_at).toLocaleDateString('en-US', {
                      year: 'numeric', month: 'short', day: 'numeric'
                    })
                    : '\u2014'}
                </td>
                <td className="px-6 py-4 text-right">
                  <button
                    onClick={() => handleDeleteUser(user)}
                    disabled={user.auth_user_id === currentUser?.id || processingId === user.id}
                    className={`p-2 rounded-lg transition-colors ${
                      user.auth_user_id === currentUser?.id
                        ? 'text-stone-300 cursor-not-allowed'
                        : 'text-stone-400 hover:text-red-600 hover:bg-red-50'
                    }`}
                    title={user.auth_user_id === currentUser?.id ? "Cannot delete yourself" : "Delete user"}
                  >
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filteredUsers.length === 0 && (
          <div className="text-center py-16 text-stone-400">
            <Users size={40} className="mx-auto mb-3 opacity-30" />
            <p className="text-sm">No users found</p>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={isDeleteModalOpen}
        user={userToProcess}
        onCancel={() => { setIsDeleteModalOpen(false); setUserToProcess(null); }}
        onConfirm={confirmDelete}
      />
    </div>
  );
}

// ─────────────────────────────────────────────
// Legacy chunk browser removed from the Admin Dashboard.
// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
// Loading Skeleton
// ─────────────────────────────────────────────
function LoadingSkeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="bg-white rounded-3xl border border-stone-200 shadow-sm p-8">
      <div className="flex items-center justify-center space-x-3 text-stone-400 mb-8">
        <Loader2 size={20} className="animate-spin" />
        <span className="text-sm">Loading data...</span>
      </div>
      <div className="space-y-4">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex items-center space-x-4 animate-pulse">
            <div className="w-9 h-9 bg-stone-100 rounded-full flex-shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="h-3 bg-stone-100 rounded-full w-1/3" />
              <div className="h-2.5 bg-stone-50 rounded-full w-1/2" />
            </div>
            <div className="h-6 w-16 bg-stone-100 rounded-lg" />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Floating Upload Progress Card
// ─────────────────────────────────────────────
const STAGES: { key: UploadStage; label: string; note?: string }[] = [
  { key: 'uploading', label: 'Uploading physical file to storage...' },
  { key: 'registering', label: 'Registering document in database...' },
  { key: 'vectorizing', label: 'AI Processing (Llama-Parse)...', note: 'This may take up to 1 minute' },
];

function UploadProgressCard({ stage, filename }: { stage: UploadStage; filename: string }) {
  if (!stage) return null;

  const currentIndex = STAGES.findIndex(s => s.key === stage);
  const progress = stage === 'uploading' ? 20 : stage === 'registering' ? 50 : 80;

  return (
    <motion.div
      initial={{ opacity: 0, y: 40, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 40, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      className="fixed bottom-6 right-6 z-50 w-96 bg-white/90 backdrop-blur-md border border-stone-200 rounded-2xl shadow-2xl p-5"
    >
      {/* Header */}
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-9 h-9 rounded-xl bg-amber-100 flex items-center justify-center">
          <Loader2 size={18} className="text-amber-600 animate-spin" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-stone-800 truncate">{filename}</p>
          <p className="text-xs text-stone-400">Processing document...</p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-2 bg-stone-100 rounded-full overflow-hidden mb-4">
        <motion.div
          className="h-full bg-gradient-to-r from-amber-400 to-amber-600 rounded-full"
          initial={{ width: '0%' }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>

      {/* Stages */}
      <div className="space-y-2.5">
        {STAGES.map((s, i) => {
          const isActive = i === currentIndex;
          const isComplete = i < currentIndex;

          return (
            <div key={s.key} className="flex items-start space-x-3">
              {/* Stage indicator */}
              <div className="flex-shrink-0 mt-0.5">
                {isComplete ? (
                  <div className="w-5 h-5 rounded-full bg-emerald-100 flex items-center justify-center">
                    <CheckCircle size={12} className="text-emerald-600" />
                  </div>
                ) : isActive ? (
                  <div className="w-5 h-5 rounded-full bg-amber-100 flex items-center justify-center">
                    <Loader2 size={12} className="text-amber-600 animate-spin" />
                  </div>
                ) : (
                  <div className="w-5 h-5 rounded-full bg-stone-100 border border-stone-200" />
                )}
              </div>

              {/* Stage text */}
              <div className="min-w-0 flex-1">
                <p className={`text-xs leading-relaxed ${
                  isActive ? 'text-stone-800 font-medium' : isComplete ? 'text-stone-400' : 'text-stone-300'
                }`}>
                  {s.label}
                </p>
                {s.note && isActive && (
                  <p className="text-[10px] text-amber-600 mt-0.5">{s.note}</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}

// ─────────────────────────────────────────────
// Notification Modal (Success / Error)
// ─────────────────────────────────────────────
function NotificationModal({
  result,
  onDismiss,
  onRetry,
  onViewFiles,
}: {
  result: UploadResult | null;
  onDismiss: () => void;
  onRetry?: () => void;
  onViewFiles?: () => void;
}) {
  if (!result) return null;

  const isSuccess = result.type === 'success';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" aria-modal="true" role="dialog">
      {/* Glassmorphic overlay */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-stone-900/30 backdrop-blur-md"
        onClick={onDismiss}
      />

      {/* Modal card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        className="relative bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl border border-stone-200 w-full max-w-md p-8"
      >
        {/* Close button */}
        <button
          onClick={onDismiss}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-stone-300 hover:text-stone-600 hover:bg-stone-100 transition-colors"
        >
          <X size={16} />
        </button>

        {/* Icon */}
        <div className="flex justify-center mb-5">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 500, damping: 20, delay: 0.1 }}
            className={`w-16 h-16 rounded-full flex items-center justify-center ${
              isSuccess ? 'bg-emerald-100' : 'bg-red-100'
            }`}
          >
            {isSuccess ? (
              <CheckCircle size={32} className="text-emerald-600" />
            ) : (
              <AlertTriangle size={32} className="text-red-600" />
            )}
          </motion.div>
        </div>

        {/* Heading */}
        <h3 className="text-xl font-semibold text-stone-900 text-center mb-2">
          {isSuccess ? 'Document Registered Successfully' : 'Upload Failed'}
        </h3>

        {/* Description */}
        <p className="text-sm text-stone-500 text-center leading-relaxed mb-6">
          {isSuccess ? (
            <>
              <span className="font-semibold text-stone-800">{result.filename}</span> has been uploaded and registered in the file manager.
            </>
          ) : (
            result.message
          )}
        </p>

        {/* Error filename */}
        {!isSuccess && (
          <div className="flex justify-center mb-6">
            <span className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-stone-50 text-stone-500 border border-stone-200">
              File: {result.filename}
            </span>
          </div>
        )}

        {/* Buttons */}
        <div className="flex space-x-3">
          {isSuccess ? (
            <>
              <button
                onClick={onDismiss}
                className="flex-1 px-4 py-3 rounded-2xl text-sm font-medium text-stone-700 bg-stone-100 hover:bg-stone-200 transition-colors"
              >
                Done
              </button>
              {onViewFiles && (
                <button
                  onClick={onViewFiles}
                  className="flex-1 px-4 py-3 rounded-2xl text-sm font-semibold text-white bg-amber-600 hover:bg-amber-700 transition-colors shadow-sm"
                >
                  View File Manager
                </button>
              )}
            </>
          ) : (
            <>
              <button
                onClick={onDismiss}
                className="flex-1 px-4 py-3 rounded-2xl text-sm font-medium text-stone-700 bg-stone-100 hover:bg-stone-200 transition-colors"
              >
                Dismiss
              </button>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="flex-1 px-4 py-3 rounded-2xl text-sm font-semibold text-white bg-red-600 hover:bg-red-700 transition-colors shadow-sm"
                >
                  Retry Upload
                </button>
              )}
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}

// ─────────────────────────────────────────────
// File Manager — Registry-First Upload with Status Tracking
// ─────────────────────────────────────────────
function FileManager() {
  const { user } = useAuth();
  const [files, setFiles] = useState<RegistryDocument[]>([]);
  const [loading, setLoading] = useState(true);

  // Upload progress state
  const [uploadStage, setUploadStage] = useState<UploadStage>(null);
  const [uploadFilename, setUploadFilename] = useState('');
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [lastFile, setLastFile] = useState<File | null>(null);

  const isVectorizing = uploadStage === 'vectorizing';

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    setLoading(true);
    const { data, error } = await supabase
      .from('compliance_documents')
      .select('*')
      .order('created_at', { ascending: false });

    if (!error) setFiles(data || []);
    setLoading(false);
  };

  const getAuthHeaders = async (): Promise<HeadersInit> => {
    const { data: { session } } = await supabase.auth.getSession();
    const headers: HeadersInit = {};
    if (session?.access_token) {
      headers['Authorization'] = `Bearer ${session.access_token}`;
    }
    return headers;
  };

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !user) return;

    setLastFile(file);
    setUploadFilename(file.filename || file.name);
    setUploadResult(null);

    // Stage 1: Uploading
    setUploadStage('uploading');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', user.id);
    formData.append('document_type', 'regulation');

    try {
      const authHeaders = await getAuthHeaders();

      // Simulate slight delay for UX (file is being sent over network)
      await new Promise(r => setTimeout(r, 600));

      // Stage 2: Registering
      setUploadStage('registering');
      await new Promise(r => setTimeout(r, 400));

      // Stage 3: Vectorizing (the long part — the backend does all 3 internally)
      setUploadStage('vectorizing');

      const response = await fetch(`${API_URL}/admin/upload-pdf`, {
        method: 'POST',
        headers: authHeaders,
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.status === 'success') {
        setUploadResult({
          type: 'success',
          filename: file.name,
          message: result.message,
        });
      } else if (result.status === 'partial_success') {
        setUploadResult({
          type: 'error',
          filename: file.name,
          message: result.message || 'File registered, but AI processing failed. Please delete it and upload again.',
        });
      } else {
        setUploadResult({
          type: 'error',
          filename: file.name,
          message: result.message || 'An unknown error occurred during upload.',
        });
      }
    } catch (err: any) {
      setUploadResult({
        type: 'error',
        filename: file.name,
        message: err.message || 'Failed to connect to the backend server.',
      });
    } finally {
      setUploadStage(null);
      event.target.value = '';
      fetchFiles();
    }
  }, [user]);

  const handleRetry = useCallback(() => {
    setUploadResult(null);
    // Trigger the file input again
    const input = document.getElementById('fm-file-input') as HTMLInputElement;
    if (input) input.click();
  }, []);

  const handleDeleteFile = async (filename: string, docId: string) => {
    if (!confirm(`Warning: This will delete the file, the registry record, AND all AI vectors for "${filename}". Continue?`)) return;

    try {
      const authHeaders = await getAuthHeaders();
      const response = await fetch(
        `${API_URL}/admin/delete-document/${docId}?filename=${encodeURIComponent(filename)}`,
        { method: 'DELETE', headers: authHeaders }
      );

      if (response.ok) {
        setFiles(prev => prev.filter(f => f.id !== docId));
      }
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  const getStatusBadge = (status?: string) => {
    const s = status || 'Completed';
    if (s === 'Completed') {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200">
          <CheckCircle size={11} /> Ready
        </span>
      );
    }
    if (s === 'Vectorizing...') {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-lg bg-amber-50 text-amber-700 border border-amber-200 animate-pulse">
          <Loader2 size={11} className="animate-spin" /> Vectorizing
        </span>
      );
    }
    if (s.toLowerCase().includes('error')) {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-lg bg-red-50 text-red-700 border border-red-200">
          <AlertTriangle size={11} /> Failed
        </span>
      );
    }
    return (
      <span className="inline-flex items-center text-xs font-medium px-2.5 py-1 rounded-lg bg-stone-50 text-stone-500 border border-stone-200">
        {s}
      </span>
    );
  };

  if (loading) {
    return <LoadingSkeleton rows={4} />;
  }

  return (
    <>
      <div className="space-y-6">
        {/* Header with Upload Button */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-stone-800">File Manager</h3>
            <p className="text-sm text-stone-500 mt-0.5">{files.length} document{files.length !== 1 ? 's' : ''} registered</p>
          </div>
          <label className={`flex items-center space-x-2 bg-amber-600 hover:bg-amber-700 text-white rounded-xl px-4 py-2.5 text-sm font-medium transition-all shadow-sm cursor-pointer ${uploadStage ? 'opacity-50 cursor-wait pointer-events-none' : ''}`}>
            <Upload size={16} />
            <span>Upload PDF</span>
            <input
              id="fm-file-input"
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleFileUpload}
              disabled={!!uploadStage}
            />
          </label>
        </div>

        {/* Table or Empty State */}
        {files.length === 0 && !isVectorizing ? (
          <div className="text-center py-20 bg-white rounded-3xl border border-stone-200 shadow-sm">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-stone-100 flex items-center justify-center">
              <FileText size={28} className="text-stone-300" />
            </div>
            <p className="text-stone-600 font-medium text-lg">No documents registered in the system</p>
            <p className="text-stone-400 text-sm mt-1.5 max-w-sm mx-auto">
              Upload a PDF to register it in the file manager. BridgeAI will prepare it for AI retrieval automatically.
            </p>
          </div>
        ) : (
          <div className="relative">
            {/* Busy overlay during vectorization */}
            <AnimatePresence>
              {isVectorizing && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 z-10 bg-white/60 backdrop-blur-[2px] rounded-3xl flex items-center justify-center"
                >
                  <div className="flex items-center space-x-3 bg-white/90 backdrop-blur-md border border-stone-200 rounded-2xl px-6 py-4 shadow-lg">
                    <Loader2 size={18} className="text-amber-600 animate-spin" />
                    <p className="text-sm font-medium text-stone-700">Updating File Manager... file will appear shortly.</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="bg-white rounded-3xl border border-stone-200 shadow-sm overflow-hidden">
              <table className="w-full text-sm text-left">
                <thead>
                  <tr className="border-b border-stone-100">
                    <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">File Name</th>
                    <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Type</th>
                    <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Uploaded At</th>
                    <th className="text-left px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Status</th>
                    <th className="text-right px-6 py-3.5 text-xs font-semibold text-stone-400 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-50">
                  {files.map(file => (
                    <tr key={file.id} className="hover:bg-stone-50/50 transition-colors group">
                      <td className="px-6 py-4">
                        <span className="font-medium text-stone-800">{file.document_name}</span>
                      </td>
                      <td className="px-6 py-4">
                        <span className="text-xs font-medium text-stone-500 bg-stone-100 px-2 py-0.5 rounded capitalize">{file.document_type}</span>
                      </td>
                      <td className="px-6 py-4 text-stone-600">
                        {new Date(file.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}
                      </td>
                      <td className="px-6 py-4">
                        {getStatusBadge(file.status)}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => handleDeleteFile(file.document_name, file.id)}
                          className="p-2 rounded-lg text-stone-300 hover:text-red-600 hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
                          title="Delete document"
                        >
                          <Trash2 size={16} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Floating Progress Card */}
      <AnimatePresence>
        {uploadStage && (
          <UploadProgressCard stage={uploadStage} filename={uploadFilename} />
        )}
      </AnimatePresence>

      {/* Success / Error Modal */}
      <AnimatePresence>
        {uploadResult && (
          <NotificationModal
            result={uploadResult}
            onDismiss={() => {
              setUploadResult(null);
              fetchFiles();
            }}
            onRetry={uploadResult.type === 'error' ? handleRetry : undefined}
            onViewFiles={uploadResult.type === 'success' ? () => {
              setUploadResult(null);
              fetchFiles();
            } : undefined}
          />
        )}
      </AnimatePresence>
    </>
  );
}

// ─────────────────────────────────────────────
// Delete Confirmation Modal
// ─────────────────────────────────────────────
interface DeleteConfirmModalProps {
  isOpen: boolean;
  user: UserProfile | null;
  onCancel: () => void;
  onConfirm: () => void;
}

function DeleteConfirmModal({ isOpen, user, onCancel, onConfirm }: DeleteConfirmModalProps) {
  if (!isOpen || !user) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      aria-modal="true"
      role="dialog"
    >
      {/* Blurred overlay */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-stone-900/40 backdrop-blur-md"
        onClick={onCancel}
      />

      {/* Modal card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        className="relative bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl border border-stone-200 w-full max-w-md p-8"
      >
        {/* Red warning icon */}
        <div className="flex justify-center mb-5">
          <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center">
            <AlertTriangle size={32} className="text-red-600" />
          </div>
        </div>

        {/* Heading */}
        <h3 className="text-xl font-semibold text-stone-900 text-center mb-2">
          Permanently delete user?
        </h3>

        {/* Description */}
        <p className="text-sm text-stone-500 text-center leading-relaxed mb-1">
          You are about to permanently delete{' '}
          <span className="font-semibold text-stone-800">
            {user.full_name || user.email}
          </span>
          &apos;s account.
        </p>
        <p className="text-xs text-red-500 text-center mb-7">
          This will remove them from authentication and all associated data. This action cannot be undone.
        </p>

        {/* User info card */}
        <div className="flex items-center space-x-3 bg-stone-50 border border-stone-200 rounded-2xl px-4 py-3 mb-7">
          <div className="w-9 h-9 rounded-full bg-stone-200 flex items-center justify-center text-stone-600 uppercase font-bold text-xs flex-shrink-0">
            {(user.full_name?.[0] || user.email?.[0] || 'U').toUpperCase()}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-stone-800 truncate">{user.full_name || 'Unnamed'}</p>
            <p className="text-xs text-stone-400 truncate">{user.email}</p>
          </div>
          <span className={`ml-auto text-xs font-semibold px-2 py-1 rounded-lg ${user.role === 'admin' ? 'bg-amber-100 text-amber-700' : 'bg-stone-100 text-stone-500'}`}>
            {user.role}
          </span>
        </div>

        {/* Action buttons */}
        <div className="flex space-x-3">
          <button
            onClick={onCancel}
            className="flex-1 px-4 py-3 rounded-2xl text-sm font-medium text-stone-700 bg-stone-100 hover:bg-stone-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 px-4 py-3 rounded-2xl text-sm font-semibold text-white bg-red-600 hover:bg-red-700 active:bg-red-800 transition-colors shadow-sm"
          >
            Delete Account
          </button>
        </div>
      </motion.div>
    </div>
  );
}
