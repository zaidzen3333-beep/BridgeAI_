/**
 * Prediction Page — AI Delay Prediction with chat interface + rich results dashboard.
 *
 * Left panel (50%): Conversational chat — collects shipment details.
 * Right panel (50%): Rich dashboard — Weather, Timeline, Root Cause Cards, PDF download.
 */

import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Send, Bot, User, Clock, AlertTriangle, TrendingUp,
  Package, Loader2, Anchor, Plane, Download,
  Wind, Thermometer, Droplets, MapPin,
  PanelRightClose, PanelRight
} from 'lucide-react';
import { jsPDF } from 'jspdf';
import { predictChat, PredictionResponse, getPredictionHistory } from '../lib/api';
import { useAuth } from '../lib/auth';

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────
interface ChatMessage {
  role: 'user' | 'ai';
  content: string;
}

interface WeatherData {
  severity_score: number;
  temp_c: number;
  condition_text: string;
  wind_kph: number;
  humidity: number;
}

interface ShapCause {
  stage: 'Origin' | 'Transit' | 'Destination' | 'General';
  stage_label?: string;
  days: number;
  title: string;
  detailed_cause: string;
}

interface PredictionResult {
  delay_days: number;
  shap_causes: ShapCause[];
  detailed_analysis?: string | null;
  document_warning: string | null;
  language?: string;
  ui_labels?: Record<string, string>;
  env_scores?: {
    origin_weather_data?: WeatherData;
    destination_weather_data?: WeatherData;
    origin_weather?: number;
    destination_weather?: number;
    origin_congestion?: number;
    destination_congestion?: number;
  };
  variables: {
    direction: string;
    transport_mode: string;
    weight: number;
    origin: string;
    destination: string;
  } | null;
}

interface PredictionHistory {
  id: number;
  created_at: string;
  user_message: string;
  agent_response: string;
  prediction_data: any;
}

// ─────────────────────────────────────────────
// Stage color map
// ─────────────────────────────────────────────
const STAGE_STYLES: Record<string, { bg: string; border: string; badge: string; dot: string; text: string }> = {
  Origin:      { bg: 'bg-blue-50',   border: 'border-blue-200',   badge: 'bg-blue-100 text-blue-800',    dot: 'bg-blue-500',   text: 'text-blue-700' },
  Transit:     { bg: 'bg-amber-50',  border: 'border-amber-200',  badge: 'bg-amber-100 text-amber-800',  dot: 'bg-amber-500',  text: 'text-amber-700' },
  Destination: { bg: 'bg-red-50',    border: 'border-red-200',    badge: 'bg-red-100 text-red-800',      dot: 'bg-red-500',    text: 'text-red-700' },
  General:     { bg: 'bg-stone-50',  border: 'border-stone-200',  badge: 'bg-stone-200 text-stone-700',  dot: 'bg-stone-400',  text: 'text-stone-600' },
};

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────
function WeatherCard({ label, data, congestion }: {
  label: string;
  data?: WeatherData;
  congestion?: number;
}) {
  const cong = congestion ?? 1;
  const congLabel = ['', 'Empty', 'Low', 'Normal', 'Heavy', 'Gridlock'][cong] ?? 'Unknown';
  const congColor = cong <= 2 ? 'text-green-600' : cong === 3 ? 'text-amber-600' : 'text-red-600';

  return (
    <div className="bg-white rounded-xl border border-stone-200 p-4 shadow-sm">
      <p className="text-[10px] font-semibold text-stone-400 uppercase tracking-wider mb-3 flex items-center gap-1">
        <MapPin size={10} /> {label}
      </p>
      {data ? (
        <div className="space-y-2">
          <p className="text-sm font-semibold text-stone-800">{data.condition_text}</p>
          <div className="grid grid-cols-3 gap-2">
            <div className="flex items-center gap-1.5">
              <Thermometer size={12} className="text-orange-400 flex-shrink-0" />
              <div>
                <p className="text-[9px] text-stone-400">Temp</p>
                <p className="text-xs font-semibold text-stone-700">{data.temp_c}°C</p>
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              <Wind size={12} className="text-blue-400 flex-shrink-0" />
              <div>
                <p className="text-[9px] text-stone-400">Wind</p>
                <p className="text-xs font-semibold text-stone-700">{data.wind_kph} km/h</p>
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              <Droplets size={12} className="text-sky-400 flex-shrink-0" />
              <div>
                <p className="text-[9px] text-stone-400">Humidity</p>
                <p className="text-xs font-semibold text-stone-700">{data.humidity}%</p>
              </div>
            </div>
          </div>
          <div className="flex items-center justify-between pt-1 border-t border-stone-100">
            <p className="text-[10px] text-stone-400">Port Congestion</p>
            <p className={`text-xs font-bold ${congColor}`}>{congLabel} ({cong}/5)</p>
          </div>
        </div>
      ) : (
        <p className="text-xs text-stone-400 italic">Weather data unavailable</p>
      )}
    </div>
  );
}

function TimelineNode({ stage, displayStage, days, active, dayShort }: {
  stage: string;
  displayStage?: string;
  days: number;
  active: boolean;
  dayShort?: string;
}) {
  const s = STAGE_STYLES[stage] ?? STAGE_STYLES.General;
  return (
    <div className="flex flex-col items-center gap-1">
      <div className={`w-10 h-10 rounded-full border-2 flex items-center justify-center shadow-sm transition-all ${
        active ? `${s.dot} border-transparent` : 'bg-white border-stone-300'
      }`}>
        {stage === 'Origin' && <Anchor size={16} className={active ? 'text-white' : 'text-stone-400'} />}
        {stage === 'Transit' && (active
          ? <span className="text-white text-[10px] font-bold">✈</span>
          : <span className="text-stone-400 text-[10px]">✈</span>
        )}
        {stage === 'Destination' && <MapPin size={16} className={active ? 'text-white' : 'text-stone-400'} />}
        {stage === 'General' && <Package size={14} className={active ? 'text-white' : 'text-stone-400'} />}
      </div>
      <p className={`text-[10px] font-semibold ${active ? s.text : 'text-stone-400'}`}>{displayStage || stage}</p>
      {days > 0 && <p className={`text-[10px] font-mono ${active ? s.text : 'text-stone-400'}`}>+{days}{dayShort || 'd'}</p>}
    </div>
  );
}

// ─────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────
export default function Prediction() {
  const [searchParams, setSearchParams] = useSearchParams();
  const predictionId = searchParams.get('id');
  const { user } = useAuth();

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'ai',
      content:
        "Hello! I'm the BridgeAI Delay Prediction Assistant. I can predict customs clearance delays for your shipments.\n\nTo get started, please tell me about your shipment. I'll need:\n1. **Direction** (Import or Export)\n2. **Transport Mode** (Sea or Air)\n3. **Weight** (in kg)\n4. **Origin** (country/port)\n5. **Destination** (country/port)\n\nYou can provide all details at once or we can go step by step!",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionHistory[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const [chatWidthPercent, setChatWidthPercent] = useState(50);
  const [isDashboardVisible, setIsDashboardVisible] = useState(true);
  const [isDraggingDashboard, setIsDraggingDashboard] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingDashboard || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const newPercent = Math.max(20, Math.min((x / rect.width) * 100, 80));
      setChatWidthPercent(newPercent);
    };
    const handleMouseUp = () => {
      setIsDraggingDashboard(false);
      document.body.classList.remove('select-none');
    };

    if (isDraggingDashboard) {
      document.body.classList.add('select-none');
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.classList.remove('select-none');
    };
  }, [isDraggingDashboard]);

  // ── Clean Light-Mode PDF Report ──────────────────────────────────────
  const handleDownloadPDF = () => {
    if (!prediction || downloading) return;
    setDownloading(true);
    try {
      const pdf = new jsPDF('p', 'mm', 'a4');
      const W = pdf.internal.pageSize.getWidth();   // 210
      const H = pdf.internal.pageSize.getHeight();  // 297
      const mg = 16;   // left/right margin
      const cw = W - mg * 2;
      let y = mg;

      // ── Color helpers ────────────────────────────────────────
      const rgb = (h: string): [number,number,number] => [
        parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16),
      ];
      const fc  = (c: string) => pdf.setFillColor(...rgb(c));
      const tc  = (c: string) => pdf.setTextColor(...rgb(c));
      const dc  = (c: string) => pdf.setDrawColor(...rgb(c));
      const B   = (sz: number) => { pdf.setFont('helvetica','bold');   pdf.setFontSize(sz); };
      const N   = (sz: number) => { pdf.setFont('helvetica','normal'); pdf.setFontSize(sz); };
      const I   = (sz: number) => { pdf.setFont('helvetica','italic'); pdf.setFontSize(sz); };
      const newPage = (need = 20) => {
        if (y + need > H - 18) { pdf.addPage(); addPageHeader(); y = 38; }
      };

      // ── Repeating page header (thin accent bar) ───────────────────
      const addPageHeader = () => {
        fc('#2563eb'); pdf.rect(0, 0, W, 3, 'F'); // blue accent strip
        fc('#ffffff'); pdf.rect(0, 3, W, 22, 'F'); // white header bg
        tc('#1e3a5f'); B(11);
        pdf.text('BridgeAI — Customs Delay Prediction Report', mg, 18);
        tc('#6b7280'); N(7.5);
        pdf.text(`Generated ${new Date().toLocaleString()}`, W - mg, 18, { align: 'right' });
        // thin divider
        dc('#e5e7eb'); pdf.setLineWidth(0.2);
        pdf.line(mg, 25, W - mg, 25);
      };

      // ── Page 1 header ───────────────────────────────────────
      addPageHeader();
      y = 32;

      // ── Section heading helper ─────────────────────────────
      const section = (title: string) => {
        newPage(16);
        tc('#1e3a5f'); B(9);
        pdf.text(title, mg, y);
        dc('#2563eb'); pdf.setLineWidth(0.5);
        pdf.line(mg, y + 2, mg + 40, y + 2);
        pdf.setLineWidth(0.2);
        y += 8;
      };

      // ── Key-value row helper ──────────────────────────────
      const kv = (label: string, value: string, vColor = '#111827') => {
        newPage(7);
        tc('#6b7280'); N(8); pdf.text(label + ':', mg + 2, y);
        tc(vColor);   B(8); pdf.text(value, mg + 52, y);
        y += 6;
      };

      // ──────────────────────────────────────────────────
      // A  — PREDICTION SUMMARY HERO
      // ──────────────────────────────────────────────────
      const dd = prediction.delay_days;
      const dColor = dd <= 3 ? '#16a34a' : dd <= 7 ? '#d97706' : '#dc2626';
      const dBg    = dd <= 3 ? '#f0fdf4' : dd <= 7 ? '#fffbeb' : '#fef2f2';
      const dBorder = dd <= 3 ? '#86efac' : dd <= 7 ? '#fde68a' : '#fca5a5';
      const riskTxt = dd <= 3 ? 'LOW RISK — Expedited clearance likely'
        : dd <= 7 ? 'MODERATE — Standard processing time'
        : 'HIGH RISK — Significant bottlenecks detected';

      // Hero card
      fc(dBg); dc(dBorder); pdf.setLineWidth(0.4);
      pdf.roundedRect(mg, y, cw, 36, 3, 3, 'FD');
      pdf.setLineWidth(0.2);

      // Big delay number
      tc(dColor); B(42); pdf.text(`${dd}`, mg + 10, y + 26);
      tc('#9ca3af'); B(16); pdf.text('days', mg + 30, y + 26);

      // Risk label pill
      fc(dColor); pdf.roundedRect(mg + 56, y + 7, 80, 7, 2, 2, 'F');
      tc('#ffffff'); B(7); pdf.text(riskTxt, mg + 96, y + 12.5, { align: 'center' });

      // Risk gauge bar
      fc('#e5e7eb'); dc('#e5e7eb');
      pdf.roundedRect(mg + 56, y + 19, 80, 4, 1, 1, 'FD');
      fc(dColor);
      pdf.roundedRect(mg + 56, y + 19, 80 * Math.min(dd / 14, 1), 4, 1, 1, 'F');
      tc('#6b7280'); N(7);
      pdf.text(`0d`, mg + 56, y + 28);
      pdf.text(`14d max`, mg + 136, y + 28, { align: 'right' });
      y += 42;

      // ──────────────────────────────────────────────────
      // B  — SHIPMENT DETAILS TABLE
      // ──────────────────────────────────────────────────
      if (prediction.variables) {
        const v = prediction.variables;
        section('Shipment Details');

        // 2-column table
        const col1 = [
          ['Direction', v.direction],
          ['Transport Mode', v.transport_mode],
          ['Cargo Weight', `${Number(v.weight).toLocaleString()} kg`],
        ];
        const col2 = [
          ['Origin', v.origin],
          ['Destination', v.destination],
          ['Route Type', v.transport_mode === 'Sea' ? 'Maritime freight' : 'Air freight'],
        ];
        const tStart = y;
        col1.forEach(([lbl, val]) => { kv(lbl, val); });
        const midY = y;
        y = tStart;
        // Right column — offset by half the page width
        col2.forEach(([lbl, val]) => {
          newPage(7);
          tc('#6b7280'); N(8); pdf.text(lbl + ':', mg + cw/2 + 4, y);
          tc('#111827'); B(8); pdf.text(val, mg + cw/2 + 54, y);
          y += 6;
        });
        y = Math.max(midY, y) + 4;

        // Divider line between sections
        dc('#e5e7eb'); pdf.line(mg, y, W - mg, y);
        y += 6;
      }

      // ──────────────────────────────────────────────────
      // C  — LIVE WEATHER & PORT CONDITIONS
      // ──────────────────────────────────────────────────
      const env2 = prediction.env_scores ?? {};
      const owd = env2.origin_weather_data as WeatherData | undefined;
      const dwd = env2.destination_weather_data as WeatherData | undefined;
      const congLabels = ['','Empty','Low','Normal','Heavy','Gridlock'];

      if (owd || dwd) {
        section('Live Weather & Port Conditions');
        const halfW = (cw - 4) / 2;

        const wxCard = (xOff: number, loc: string, wd?: WeatherData, cong?: number) => {
          newPage(40);
          fc('#f9fafb'); dc('#e5e7eb'); pdf.setLineWidth(0.3);
          pdf.roundedRect(mg + xOff, y, halfW, 34, 2, 2, 'FD');
          pdf.setLineWidth(0.2);

          // Location name
          tc('#1e3a5f'); B(8.5);
          pdf.text(loc, mg + xOff + 4, y + 7);

          if (wd) {
            // Condition
            tc('#374151'); N(8);
            pdf.text(wd.condition_text, mg + xOff + 4, y + 14);

            // Stats row
            const stats = [
              `${wd.temp_c}°C`,
              `${wd.wind_kph} km/h`,
              `${wd.humidity}% hum.`,
            ];
            stats.forEach((s, i) => {
              tc('#6b7280'); N(7.5);
              pdf.text(s, mg + xOff + 4 + i * (halfW / 3), y + 21);
            });

            // Congestion badge
            const cl = cong ?? 1;
            const cbg = cl <= 2 ? '#d1fae5' : cl <= 3 ? '#fef3c7' : '#fee2e2';
            const ctc = cl <= 2 ? '#065f46' : cl <= 3 ? '#92400e' : '#991b1b';
            fc(cbg); dc(cbg);
            pdf.roundedRect(mg + xOff + 4, y + 26, halfW - 8, 5, 1, 1, 'F');
            tc(ctc); B(7);
            pdf.text(`Port: ${congLabels[cl] ?? 'Unknown'} (${cl}/5)`, mg + xOff + halfW/2, y + 29.5, { align: 'center' });
          } else {
            tc('#9ca3af'); I(8); pdf.text('Data unavailable', mg + xOff + 4, y + 18);
          }
        };

        wxCard(0,            `Origin: ${prediction.variables?.origin ?? ''}`,      owd, env2.origin_congestion);
        wxCard(halfW + 4,    `Destination: ${prediction.variables?.destination ?? ''}`, dwd, env2.destination_congestion);
        y += 38;
      }

      // ──────────────────────────────────────────────────
      // D  — DOCUMENT ASSUMPTION NOTICE
      // ──────────────────────────────────────────────────
      if (prediction.document_warning) {
        section(label('document_assumption', 'Document Assumption'));
        newPage(20);
        fc('#fffbeb'); dc('#fcd34d'); pdf.setLineWidth(0.4);
        const dwLines = pdf.splitTextToSize(prediction.document_warning, cw - 10) as string[];
        const dwH = 10 + dwLines.length * 5;
        pdf.roundedRect(mg, y, cw, dwH, 2, 2, 'FD');
        pdf.setLineWidth(0.2);
        tc('#92400e'); B(8); pdf.text('⚠  Assumption Notice', mg + 4, y + 7);
        tc('#78350f'); N(8);
        dwLines.forEach((line: string, i: number) => pdf.text(line, mg + 4, y + 13 + i * 5));
        y += dwH + 6;
      }

      // ──────────────────────────────────────────────────
      // E  — STAGE TIMELINE SUMMARY
      // ──────────────────────────────────────────────────
      if (prediction.shap_causes.length > 0) {
        const stageColors2: Record<string,[string,string]> = {
          Origin:      ['#dbeafe','#1d4ed8'],
          Transit:     ['#fef3c7','#92400e'],
          Destination: ['#fee2e2','#991b1b'],
          General:     ['#f3f4f6','#374151'],
        };

        // Group by stage
        const stages = ['Origin','Transit','Destination','General'];
        const grouped: Record<string,{days:number}[]> = {};
        stages.forEach(s => { grouped[s] = prediction.shap_causes.filter(c=>c.stage===s); });
        const stageTotals = stages.map(s => ({
          stage: s,
          days: grouped[s].reduce((a,c)=>a+c.days,0),
        })).filter(s=>s.days>0);

        section(label('delay_timeline', 'Delay Timeline'));
        newPage(12 + stageTotals.length * 10);

        // Horizontal stacked bar showing proportion of delay per stage
        const totalDays = stageTotals.reduce((a,s)=>a+s.days,0) || 1;
        const barTotalW = cw;
        let barX2 = mg;
        stageTotals.forEach(({ stage, days }) => {
          const [bg, tx] = stageColors2[stage] ?? ['#f3f4f6','#374151'];
          const bw = (days / totalDays) * barTotalW;
          fc(bg); dc(bg); pdf.rect(barX2, y, bw, 8, 'FD');
          if (bw > 20) {
            tc(tx); B(7);
            pdf.text(`${stageLabel(stage)} +${days}${label('day_short', 'd')}`, barX2 + bw/2, y + 5.5, { align: 'center' });
          }
          barX2 += bw;
        });
        y += 12;

        // Per-stage detail rows
        stageTotals.forEach(({ stage, days }) => {
          const [bg, tx] = stageColors2[stage] ?? ['#f3f4f6','#374151'];
          newPage(8);
          fc(bg); dc(bg);
          pdf.roundedRect(mg, y, 28, 6, 1, 1, 'F');
          tc(tx); B(7);
          pdf.text(stageLabel(stage).toUpperCase(), mg + 14, y + 4.5, { align: 'center' });
          tc('#111827'); N(8);
          pdf.text(`+${days.toFixed(1)} ${label('days', 'days')}`, mg + 32, y + 4.5);
          y += 8;
        });
        y += 4;

        // ──────────────────────────────────────────────────
        // F  — ROOT CAUSE ANALYSIS CARDS
        // ──────────────────────────────────────────────────
        section(label('root_cause_analysis', 'Root Cause Analysis'));

        prediction.shap_causes.forEach((cause, idx) => {
          const [bg, tx] = stageColors2[cause.stage] ?? ['#f3f4f6','#374151'];
          const detailLines = pdf.splitTextToSize(cause.detailed_cause, cw - 40) as string[];
          const cardH = 14 + detailLines.length * 4.8;
          newPage(cardH + 4);

          // Card background — white with left accent
          fc('#ffffff'); dc('#e5e7eb'); pdf.setLineWidth(0.2);
          pdf.roundedRect(mg, y, cw, cardH, 2, 2, 'FD');
          // Left accent stripe
          fc(tx); pdf.rect(mg, y, 3, cardH, 'F');

          // Stage pill
          fc(bg); pdf.roundedRect(mg + 7, y + 3, 24, 5.5, 1, 1, 'F');
          tc(tx); B(6.5);
          pdf.text((cause.stage_label || stageLabel(cause.stage)).toUpperCase(), mg + 19, y + 7, { align: 'center' });

          // Title
          tc('#111827'); B(8.5);
          pdf.text(cause.title, mg + 35, y + 7);

          // Days badge (right side)
          const daysTxt = `+${cause.days.toFixed(1)}${label('day_short', 'd')}`;
          fc(tx); pdf.roundedRect(W - mg - 20, y + 2, 18, 7, 1, 1, 'F');
          tc('#ffffff'); B(8);
          pdf.text(daysTxt, W - mg - 11, y + 7, { align: 'center' });

          // Detail text
          tc('#4b5563'); N(7.5);
          detailLines.forEach((line: string, i: number) => {
            pdf.text(line, mg + 7, y + 13 + i * 4.8);
          });

          // Mini proportion bar
          const barPct = Math.min(cause.days / dd, 1);
          fc('#e5e7eb'); pdf.rect(mg + 7, y + cardH - 4, cw - 14, 2, 'F');
          fc(tx);       pdf.rect(mg + 7, y + cardH - 4, (cw - 14) * barPct, 2, 'F');

          y += cardH + 4;
        });
      }

      // ── Footer on every page ────────────────────────────────────
      const totalPages = (pdf as any).internal.getNumberOfPages();
      for (let p = 1; p <= totalPages; p++) {
        pdf.setPage(p);
        fc('#f9fafb'); pdf.rect(0, H - 10, W, 10, 'F');
        dc('#e5e7eb'); pdf.setLineWidth(0.2); pdf.line(mg, H - 10, W - mg, H - 10);
        tc('#9ca3af'); N(7);
        pdf.text('CONFIDENTIAL — BridgeAI Customs Intelligence Platform', W/2, H - 3.5, { align: 'center' });
        pdf.text(`Page ${p} of ${totalPages}`, W - mg, H - 3.5, { align: 'right' });
      }

      pdf.save('BridgeAI_Prediction_Report.pdf');
    } catch (err: any) {
      console.error('PDF generation failed:', err);
    } finally {
      setDownloading(false);
    }
  };

  const fetchHistoryList = async () => {
    if (!user) return;
    setIsLoadingHistory(true);
    try {
      const result = await getPredictionHistory(user.id);
      if (result.status === 'success' && result.data) {
        setHistory(result.data);
        window.dispatchEvent(new Event('predictionHistoryUpdated'));
      }
    } catch (err) {
      console.error('Error fetching prediction history list:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => { fetchHistoryList(); }, []);

  // Sync with ?id= URL param
  useEffect(() => {
    const loadById = async () => {
      if (!predictionId || !user) return;
      let pool = history;
      if (!pool.length) {
        try {
          const res = await getPredictionHistory(user.id);
          if (res.status === 'success' && res.data) { setHistory(res.data); pool = res.data; }
        } catch { /* ignore */ }
      }
      const found = pool.find((h) => String(h.id) === predictionId);
      if (found) {
        // If chat history is stored in prediction_data, restore the whole conversation
        if (found.prediction_data && found.prediction_data.chat_history) {
          setMessages(found.prediction_data.chat_history as ChatMessage[]);
        } else {
          // Fallback to legacy single-message behavior
          setMessages([
            { role: 'user', content: found.user_message },
            { role: 'ai', content: found.agent_response },
          ]);
        }
        setPrediction(found.prediction_data);
      }
    };
    loadById();
  }, [predictionId]);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);
    try {
      const historyPayload = messages.slice(1).map((m) => ({ role: m.role, content: m.content }));
      const pId = predictionId ? Number(predictionId) : null;
      
      const response: PredictionResponse = await predictChat(historyPayload, userMessage, user?.id || '', prediction as any, pId);
      
      setMessages((prev) => [...prev, { role: 'ai', content: response.message }]);
      
      if (response.status === 'success' && response.prediction_data) {
        setPrediction(response.prediction_data as any);
      }
      
      // If this is a new conversation, update the URL so follow-up messages go to the same ID
      if (response.prediction_id && !predictionId) {
        setSearchParams({ id: String(response.prediction_id) });
      }
      
      fetchHistoryList();
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : 'Unknown error';
      setMessages((prev) => [...prev, { role: 'ai', content: `Sorry, an error occurred: ${msg}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleReset = () => {
    setSearchParams({});
    setMessages([{
      role: 'ai',
      content: "Let's start a new prediction! Tell me about your shipment — Direction, Transport Mode, Weight, Origin, and Destination.",
    }]);
    setPrediction(null);
  };

  // ── Derived timeline data ──
  const stageGroups = prediction ? ['Origin', 'Transit', 'Destination', 'General'].reduce((acc, stage) => {
    const causes = prediction.shap_causes.filter((c) => c.stage === stage);
    const total = causes.reduce((s, c) => s + c.days, 0);
    acc[stage] = { causes, total: Math.round(total * 10) / 10 };
    return acc;
  }, {} as Record<string, { causes: ShapCause[]; total: number }>) : null;

  const env = prediction?.env_scores ?? {};
  const originWeatherData = env.origin_weather_data as WeatherData | undefined;
  const destWeatherData = env.destination_weather_data as WeatherData | undefined;
  const uiLabels = prediction?.ui_labels ?? {};
  const label = (key: string, fallback: string) => uiLabels[key] || fallback;
  const stageLabel = (stage: string) => label(stage.toLowerCase(), stage);

  return (
    <div ref={containerRef} className="flex h-full w-full overflow-hidden relative">
      {/* ── LEFT: Chat Panel ── */}
      <div
        style={{ width: isDashboardVisible ? `${chatWidthPercent}%` : '100%' }}
        className="flex flex-col border-r border-stone-200/60 min-w-0 transition-all duration-300 relative"
      >
        {isDashboardVisible && (
          <div
            onMouseDown={(e) => {
              e.preventDefault();
              setIsDraggingDashboard(true);
            }}
            className={`absolute top-0 right-[-3px] w-[6px] h-full cursor-col-resize z-10 transition-colors ${
              isDraggingDashboard ? 'bg-amber-400' : 'hover:bg-amber-400/50'
            }`}
          />
        )}
        <div className="px-6 py-4 border-b border-stone-200/60 bg-white/80 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-stone-800 flex items-center gap-2">
                <Clock size={20} className="text-amber-600" />
                AI Delay Prediction
              </h2>
              <p className="text-xs text-stone-400 mt-0.5">Tell me your shipment details and I'll predict the clearance delay</p>
            </div>
            <div className="flex items-center gap-2">
              {prediction && (
                <button onClick={handleReset} className="text-xs px-3 py-1.5 bg-stone-100 hover:bg-stone-200 rounded-lg text-stone-600 transition-colors">
                  New Prediction
                </button>
              )}
              {!isDashboardVisible && (
                <button
                  onClick={() => setIsDashboardVisible(true)}
                  className="p-1.5 text-stone-400 hover:text-stone-600 hover:bg-stone-100 rounded transition-colors"
                  title="Show Dashboard"
                >
                  <PanelRight size={18} />
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6 scrollbar-thin scrollbar-thumb-stone-200">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'ai' && (
                <div className="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot size={16} className="text-amber-700" />
                </div>
              )}
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed bg-stone-100 text-stone-700`}>
                {msg.content.split('\n').map((line, li) => (
                  <p key={li} className={`text-inherit ${li > 0 ? 'mt-1.5' : ''}`}>
                    {line.split(/(\*\*.*?\*\*)/).map((part, pi) =>
                      part.startsWith('**') && part.endsWith('**')
                        ? <strong key={pi} className="font-semibold text-inherit">{part.slice(2, -2)}</strong>
                        : <span key={pi}>{part}</span>
                    )}
                  </p>
                ))}
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-stone-800 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User size={16} className="text-stone-100" />
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center flex-shrink-0">
                <Bot size={16} className="text-amber-700" />
              </div>
              <div className="bg-stone-100 rounded-2xl px-4 py-3 flex items-center gap-2">
                <Loader2 size={14} className="animate-spin text-amber-600" />
                <span className="text-sm text-stone-500">Analyzing shipment...</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="px-6 py-4 border-t border-stone-200/60 bg-white/80 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your shipment..."
              className="flex-1 bg-stone-100 rounded-xl px-4 py-3 text-sm text-stone-800 placeholder:text-stone-400 outline-none focus:ring-2 focus:ring-amber-200 transition-all"
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="w-10 h-10 rounded-xl bg-stone-800 text-stone-50 flex items-center justify-center hover:bg-stone-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* ── RIGHT: Dashboard Panel ── */}
      {isDashboardVisible && (
        <div className="flex-1 min-w-0 flex-shrink-0 overflow-y-auto bg-stone-50/50 scrollbar-thin scrollbar-thumb-stone-200 relative pt-12">
          {/* Hide Dashboard Button */}
          <div className="absolute top-4 right-4 z-50">
            <button
              onClick={() => setIsDashboardVisible(false)}
              className="p-1.5 text-stone-400 hover:text-stone-600 hover:bg-white rounded border border-transparent hover:border-stone-200 transition-all shadow-sm"
              title="Hide Dashboard"
            >
              <PanelRightClose size={18} />
            </button>
          </div>

          {/* ── SKELETON LOADING STATE ── */}
        {loading && !prediction ? (
          <div className="p-5 space-y-4 animate-pulse">
            {/* Skeleton header row */}
            <div className="flex justify-end">
              <div className="h-7 w-32 bg-stone-200 rounded-lg" />
            </div>

            {/* Skeleton delay card */}
            <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm space-y-3">
              <div className="flex justify-between items-center">
                <div className="h-3 w-24 bg-stone-200 rounded-full" />
                <div className="h-4 w-4 bg-stone-200 rounded-full" />
              </div>
              <div className="flex items-end gap-2">
                <div className="h-10 w-16 bg-stone-200 rounded-lg" />
                <div className="h-5 w-10 bg-stone-100 rounded" />
              </div>
              <div className="h-2 bg-stone-200 rounded-full" />
              <div className="h-3 w-48 bg-stone-100 rounded-full" />
            </div>

            {/* Skeleton timeline */}
            <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm space-y-4">
              <div className="h-3 w-28 bg-stone-200 rounded-full" />
              <div className="flex items-center justify-between">
                {[0,1,2].map(i => (
                  <div key={i} className="flex flex-col items-center gap-2">
                    <div className="w-10 h-10 rounded-full bg-stone-200" />
                    <div className="h-2 w-12 bg-stone-200 rounded-full" />
                    <div className="h-2 w-8 bg-stone-100 rounded-full" />
                  </div>
                ))}
              </div>
              <div className="space-y-2">
                {[0,1,2].map(i => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="h-2 w-16 bg-stone-200 rounded-full" />
                    <div className="flex-1 h-1.5 bg-stone-200 rounded-full" />
                    <div className="h-2 w-8 bg-stone-200 rounded-full" />
                  </div>
                ))}
              </div>
            </div>

            {/* Skeleton weather */}
            <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm space-y-3">
              <div className="h-3 w-36 bg-stone-200 rounded-full" />
              <div className="grid grid-cols-2 gap-3">
                {[0,1].map(i => (
                  <div key={i} className="border border-stone-200 rounded-xl p-4 space-y-2">
                    <div className="h-2.5 w-20 bg-stone-200 rounded-full" />
                    <div className="h-3.5 w-28 bg-stone-200 rounded" />
                    <div className="grid grid-cols-3 gap-1">
                      {[0,1,2].map(j => <div key={j} className="h-6 bg-stone-100 rounded" />)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Skeleton cause cards */}
            <div className="space-y-3">
              <div className="h-3 w-36 bg-stone-200 rounded-full" />
              {[0,1,2].map(i => (
                <div key={i} className="bg-white rounded-2xl border border-stone-200 p-4 space-y-2">
                  <div className="flex justify-between">
                    <div className="flex gap-2 items-center">
                      <div className="h-5 w-16 bg-stone-200 rounded-full" />
                      <div className="h-3.5 w-32 bg-stone-200 rounded" />
                    </div>
                    <div className="h-4 w-8 bg-stone-200 rounded" />
                  </div>
                  <div className="h-2.5 w-full bg-stone-100 rounded-full" />
                  <div className="h-2.5 w-4/5 bg-stone-100 rounded-full" />
                  <div className="h-1 w-full bg-stone-100 rounded-full mt-1" />
                </div>
              ))}
            </div>

            {/* Pulse label */}
            <div className="flex items-center justify-center gap-2 py-2">
              <Loader2 size={14} className="animate-spin text-amber-500" />
              <span className="text-xs text-stone-400">AI is analyzing your shipment…</span>
            </div>
          </div>
        ) : !prediction ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-8">
            <div className="w-16 h-16 rounded-2xl bg-stone-100 flex items-center justify-center mb-4">
              <TrendingUp size={28} className="text-stone-300" />
            </div>
            <h3 className="text-stone-500 font-medium mb-1">Prediction Dashboard</h3>
            <p className="text-stone-400 text-sm leading-relaxed max-w-[280px]">
              Once you provide all shipment details in the chat, the full delay analysis will appear here.
            </p>
          </div>
        ) : (
          <div className="p-5 space-y-4">
            {/* Download button */}
            <div className="flex justify-end">
              <button
                onClick={handleDownloadPDF}
                disabled={downloading}
                className="flex items-center gap-2 text-xs px-3 py-1.5 bg-stone-800 hover:bg-stone-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-stone-50 transition-colors shadow-sm"
              >
                {downloading
                  ? <><Loader2 size={13} className="animate-spin" /><span>{label('generating', 'Generating...')}</span></>
                  : <><Download size={13} /><span>{label('download_report', 'Download Report')}</span></>
                }
              </button>
            </div>

            {/* ── Predicted Delay Card ── */}
            <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider">{label('predicted_delay', 'Predicted Delay')}</h3>
                <Clock size={16} className="text-amber-500" />
              </div>
              <div className="flex items-end gap-2 mb-3">
                <span className="text-4xl font-bold text-stone-800">{prediction.delay_days}</span>
                <span className="text-lg text-stone-400 pb-1">{label('days', 'days')}</span>
              </div>
              <div className="h-2 bg-stone-100 rounded-full overflow-hidden mb-2">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${
                    prediction.delay_days <= 3 ? 'bg-green-400' : prediction.delay_days <= 7 ? 'bg-amber-400' : 'bg-red-400'
                  }`}
                  style={{ width: `${Math.min((prediction.delay_days / 14) * 100, 100)}%` }}
                />
              </div>
              <p className="text-xs text-stone-400">
                {prediction.delay_days <= 3 ? label('low_risk', 'Low risk — expedited clearance likely')
                  : prediction.delay_days <= 7 ? label('moderate_risk', 'Moderate — standard processing time')
                  : label('high_risk', 'High risk — potential bottlenecks detected')}
              </p>
            </div>

            {/* ── Visual Timeline ── */}
            {stageGroups && (
              <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
                <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider mb-4">{label('delay_timeline', 'Delay Timeline')}</h3>
                <div className="flex items-center justify-between relative">
                  {/* Connector line */}
                  <div className="absolute top-5 left-5 right-5 h-0.5 bg-stone-200 z-0" />
                  {['Origin', 'Transit', 'Destination'].map((stage) => (
                    <div key={stage} className="z-10">
                      <TimelineNode
                        stage={stage}
                        displayStage={stageLabel(stage)}
                        days={stageGroups[stage]?.total ?? 0}
                        active={(stageGroups[stage]?.total ?? 0) > 0}
                        dayShort={label('day_short', 'd')}
                      />
                    </div>
                  ))}
                </div>
                {/* Bar breakdown */}
                <div className="mt-4 space-y-1.5">
                  {['Origin', 'Transit', 'Destination', 'General'].map((stage) => {
                    const total = stageGroups[stage]?.total ?? 0;
                    if (total === 0) return null;
                    const s = STAGE_STYLES[stage];
                    const pct = Math.min((total / prediction.delay_days) * 100, 100);
                    return (
                      <div key={stage} className="flex items-center gap-2">
                        <span className={`text-[10px] font-semibold w-20 ${s.text}`}>{stageLabel(stage)}</span>
                        <div className="flex-1 h-1.5 bg-stone-100 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${s.dot}`} style={{ width: `${pct}%` }} />
                        </div>
                        <span className={`text-[10px] font-mono font-bold w-8 text-right ${s.text}`}>+{total}{label('day_short', 'd')}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ── Weather Widget ── */}
            {(originWeatherData || destWeatherData) && prediction.variables && (
              <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
                <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider mb-3">{label('live_weather', 'Live Weather Conditions')}</h3>
                <div className="grid grid-cols-2 gap-3">
                  <WeatherCard
                    label={prediction.variables.origin}
                    data={originWeatherData}
                    congestion={env.origin_congestion}
                  />
                  <WeatherCard
                    label={prediction.variables.destination}
                    data={destWeatherData}
                    congestion={env.destination_congestion}
                  />
                </div>
              </div>
            )}

            {/* ── Shipment Details ── */}
            {prediction.variables && (
              <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
                <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider mb-3">{label('shipment_details', 'Shipment Details')}</h3>
                <div className="grid grid-cols-2 gap-2.5">
                  <InfoPill icon={prediction.variables.transport_mode === 'Sea' ? Anchor : Plane} label={label('mode', 'Mode')} value={prediction.variables.transport_mode} />
                  <InfoPill icon={Package} label={label('weight', 'Weight')} value={`${prediction.variables.weight} kg`} />
                  <InfoPill icon={TrendingUp} label={label('direction', 'Direction')} value={prediction.variables.direction} />
                  <InfoPill icon={MapPin} label={label('route', 'Route')} value={`${prediction.variables.origin} → ${prediction.variables.destination}`} />
                </div>
              </div>
            )}

            {/* ── Document Warning ── */}
            {prediction.document_warning && (
              <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 flex gap-3">
                <AlertTriangle size={16} className="text-amber-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-sm font-semibold text-amber-800 mb-1">{label('document_assumption', 'Document Assumption')}</h4>
                  <p className="text-xs text-amber-700 leading-relaxed">{prediction.document_warning}</p>
                </div>
              </div>
            )}

            {/* ── Root Cause Cards ── */}
            {prediction.shap_causes.length > 0 && (
              <div className="space-y-3">
                <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider px-1">{label('root_cause_analysis', 'Root Cause Analysis')}</h3>
                {prediction.shap_causes.map((cause, i) => {
                  const s = STAGE_STYLES[cause.stage] ?? STAGE_STYLES.General;
                  return (
                    <div key={i} className={`rounded-2xl border p-4 shadow-sm ${s.bg} ${s.border}`}>
                      <div className="flex items-start justify-between gap-3 mb-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${s.badge}`}>
                            {cause.stage_label || stageLabel(cause.stage)}
                          </span>
                          <h4 className={`text-sm font-semibold ${s.text}`}>{cause.title}</h4>
                        </div>
                        <span className={`text-sm font-bold font-mono flex-shrink-0 ${s.text}`}>
                          +{cause.days}{label('day_short', 'd')}
                        </span>
                      </div>
                      <p className="text-xs text-stone-600 leading-relaxed">{cause.detailed_cause}</p>
                      {/* Mini progress bar */}
                      <div className="mt-2.5 h-1 bg-white/70 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${s.dot}`}
                          style={{ width: `${Math.min((cause.days / prediction.delay_days) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Sub-component: Info Pill
// ─────────────────────────────────────────────
function InfoPill({ icon: Icon, label, value }: {
  icon: React.ComponentType<any>;
  label: string;
  value: string;
}) {
  return (
    <div className="bg-stone-50 rounded-xl px-3 py-2.5 flex items-center gap-2">
      <Icon size={14} className="text-stone-400 flex-shrink-0" />
      <div className="min-w-0">
        <p className="text-[10px] text-stone-400 uppercase tracking-wider">{label}</p>
        <p className="text-xs font-medium text-stone-700 truncate">{value}</p>
      </div>
    </div>
  );
}
