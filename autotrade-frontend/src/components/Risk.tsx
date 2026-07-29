import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom'; // Import Link
import { supabase } from '../lib/supabase';
import { AlertTriangle, CheckCircle, Info, MessageSquare, ChevronRight } from 'lucide-react';

export default function Risk() {
  const [screenings, setScreenings] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchScreenings = async () => {
      // Fetching including the new session_id column
      const { data, error } = await supabase
        .from('compliance_screenings')
        .select('*')
        .order('created_at', { ascending: false });

      if (!error) setScreenings(data || []);
      setLoading(false);
    };
    fetchScreenings();
  }, []);

  return (
    <div className="p-8 max-w-6xl mx-auto h-full overflow-y-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-serif text-stone-800">Compliance Risk Engine</h2>
        <p className="text-stone-500 text-sm">AI-detected anomalies in trade documentation</p>
      </div>

      <div className="grid gap-4">
        {screenings.map((s) => (
          <div key={s.id} className="bg-white p-6 rounded-3xl border border-stone-200 shadow-sm hover:shadow-md transition-all flex items-start space-x-5 group">
            {/* Status Icon */}
            <div className={`p-3 rounded-2xl flex-shrink-0 ${s.status === 'High Risk' ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'}`}>
              {s.status === 'High Risk' ? <AlertTriangle size={24} /> : <CheckCircle size={24} />}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h4 className="font-bold text-stone-800 text-lg leading-snug break-words" title={s.screening_type}>
                    {s.screening_type}
                  </h4>
                  <span className="text-xs text-stone-400 font-mono">
                    ID: {s.id.substring(0, 8)} • {new Date(s.created_at).toLocaleString()}
                  </span>
                </div>
                <span className={`text-[10px] px-2.5 py-1 rounded-full font-black uppercase tracking-wider ${s.status === 'High Risk' ? 'text-red-700 bg-red-100/50' : 'text-green-700 bg-green-100/50'}`}>
                  {s.status}
                </span>
              </div>

              <p className="text-sm text-stone-600 leading-relaxed line-clamp-2 mb-4 italic">
                "{s.ai_analysis_notes}"
              </p>

              {/* ACTION: The Deep Link */}
              {s.session_id && (
                <Link
                  to={`/assistant/${s.session_id}`}
                  className="inline-flex items-center space-x-2 text-amber-700 bg-amber-50 hover:bg-amber-100 px-4 py-2 rounded-xl text-sm font-medium transition-colors group-hover:translate-x-1 duration-300"
                >
                  <MessageSquare size={16} />
                  <span>View Full Audit Conversation</span>
                  <ChevronRight size={14} />
                </Link>
              )}
            </div>
          </div>
        ))}

        {screenings.length === 0 && !loading && (
          <div className="text-center py-20 bg-stone-50 rounded-3xl border border-dashed border-stone-200">
            <Info className="mx-auto mb-4 text-stone-300" size={48} />
            <p className="text-stone-500">No screening data detected yet.</p>
            <p className="text-stone-400 text-sm mt-1">Upload an invoice in the Assistant to trigger an audit.</p>
          </div>
        )}
      </div>
    </div>
  );
}
