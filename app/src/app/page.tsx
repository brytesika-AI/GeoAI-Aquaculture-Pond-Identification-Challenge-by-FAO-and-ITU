'use client';

import React, { useState, useEffect, useRef } from 'react';

interface PredictionPoint {
  id: string;
  lat: number;
  lon: number;
  type: 'train' | 'test' | 'africa';
  probability: number;
  actual_label: number | null;
  explanation: string;
  ndwi: number[];
  mndwi: number[];
  vh: number[];
  vv: number[];
}

export default function GeospatialDashboard() {
  const [data, setData] = useState<PredictionPoint[]>([]);
  const [filteredData, setFilteredData] = useState<PredictionPoint[]>([]);
  const [selectedPoint, setSelectedPoint] = useState<PredictionPoint | null>(null);
  
  // Dashboard Filters
  const [threshold, setThreshold] = useState<number>(0.5);
  const [showTrain, setShowTrain] = useState<boolean>(true);
  const [showTest, setShowTest] = useState<boolean>(true);
  const [showAfrica, setShowAfrica] = useState<boolean>(true);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [mapType, setMapType] = useState<string>('hybrid');
  
  // Stats
  const [stats, setStats] = useState({
    total: 0,
    ponds: 0,
    avgProb: 0
  });

  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<any>(null);
  const leafletMarkersGroupRef = useRef<any>(null);
  const [mapsLoaded, setMapsLoaded] = useState<boolean>(false);

  const flyToVietnam = () => {
    const map = leafletMapRef.current;
    if (map) map.setView([10.0, 105.5], 11);
  };

  const flyToAfrica = () => {
    const map = leafletMapRef.current;
    if (map) map.setView([0.20, 32.54], 6);
  };

  // Chatbot State
  const [chatMessages, setChatMessages] = useState<{ sender: 'user' | 'ai'; text: string }[]>([
    { sender: 'ai', text: 'Welcome! I am your AI Geospatial Consultant. Tap on any map point to inspect, or click the suggest buttons below for quick information.' }
  ]);
  const [userInput, setUserInput] = useState<string>('');
  
  // LLM Config
  const [apiService, setApiService] = useState<string>('huggingface'); // 'huggingface' | 'cloudflare' | 'ollama' | 'local'
  const [selectedModel, setSelectedModel] = useState<string>('google/gemma-2-9b-it');
  const [apiToken, setApiToken] = useState<string>('');
  const [cloudflareAccountId, setCloudflareAccountId] = useState<string>('');
  const [ollamaEndpoint, setOllamaEndpoint] = useState<string>('http://localhost:11434');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [showAdvancedChat, setShowAdvancedChat] = useState<boolean>(false);

  // 1. Fetch predictions.json on mount
  useEffect(() => {
    fetch('/predictions.json')
      .then(res => res.json())
      .then((json: PredictionPoint[]) => {
        setData(json);
        setFilteredData(json);
      })
      .catch(err => {
        console.error("Failed to load predictions.json", err);
      });
  }, []);

  // 2. Filter data and update stats when filters change
  useEffect(() => {
    const filtered = data.filter(pt => {
      if (pt.type === 'train' && !showTrain) return false;
      if (pt.type === 'test' && !showTest) return false;
      if (pt.type === 'africa' && !showAfrica) return false;
      return true;
    });
    setFilteredData(filtered);

    // Compute stats
    const total = filtered.length;
    const ponds = filtered.filter(pt => pt.probability >= threshold).length;
    const avgProb = total > 0 ? filtered.reduce((acc, pt) => acc + pt.probability, 0) / total : 0;
    
    setStats({
      total,
      ponds,
      avgProb
    });
  }, [data, threshold, showTrain, showTest, showAfrica]);

  // Update default models when service changes
  useEffect(() => {
    if (apiService === 'huggingface') {
      setSelectedModel('google/gemma-2-9b-it');
    } else if (apiService === 'cloudflare') {
      setSelectedModel('@cf/meta/llama-3-8b-instruct');
    } else if (apiService === 'ollama') {
      setSelectedModel('gemma2');
    }
  }, [apiService]);

  // 3. Load Leaflet CSS and JS dynamically (Client-side only)
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Load Leaflet CSS
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
    link.integrity = 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=';
    link.crossOrigin = '';
    document.head.appendChild(link);

    // Load Leaflet JS
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.integrity = 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=';
    script.crossOrigin = '';
    script.async = true;
    script.onload = () => {
      setMapsLoaded(true);
    };
    document.head.appendChild(script);

    return () => {
      document.head.removeChild(link);
      document.head.removeChild(script);
    };
  }, []);

  // 4. Initialize Map
  useEffect(() => {
    if (!mapsLoaded || !mapRef.current || filteredData.length === 0) return;

    const L = (window as any).L;
    if (!L) return;

    // Smart centering logic based on active datasets
    let centerLat = 10.0;
    let centerLon = 105.5;
    let defaultZoom = 11;

    const hasVietnam = filteredData.some(pt => pt.type === 'train' || pt.type === 'test');
    const hasAfrica = filteredData.some(pt => pt.type === 'africa');

    if (hasVietnam && hasAfrica) {
      // Wide view showing both regions
      centerLat = 5.0;
      centerLon = 68.0;
      defaultZoom = 3;
    } else if (hasAfrica) {
      // Focus on East Africa
      centerLat = 0.20;
      centerLon = 32.54;
      defaultZoom = 6;
    } else if (hasVietnam) {
      const lats = filteredData.map(pt => pt.lat);
      const lons = filteredData.map(pt => pt.lon);
      centerLat = (Math.max(...lats) + Math.min(...lats)) / 2;
      centerLon = (Math.max(...lons) + Math.min(...lons)) / 2;
      defaultZoom = 11;
    }

    // Initialize Leaflet map
    const map = L.map(mapRef.current, {
      zoomControl: true,
      attributionControl: true
    }).setView([centerLat, centerLon], defaultZoom);
    
    leafletMapRef.current = map;

    // Apply base tile layer
    let tileLayer;
    if (mapType === 'hybrid') {
      tileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
      });
    } else {
      tileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
      });
    }
    tileLayer.addTo(map);

    // Create a layers group for markers/heatmap
    const markersGroup = L.layerGroup().addTo(map);
    leafletMarkersGroupRef.current = markersGroup;

    // Initial draw
    updateMapElements(L, map, markersGroup);

    return () => {
      map.remove();
      leafletMapRef.current = null;
      leafletMarkersGroupRef.current = null;
    };
  }, [mapsLoaded, filteredData, mapType]);

  // 5. Update markers/layers when threshold or heatmap filters change
  useEffect(() => {
    const L = (window as any).L;
    const map = leafletMapRef.current;
    const markersGroup = leafletMarkersGroupRef.current;
    if (!L || !map || !markersGroup) return;

    updateMapElements(L, map, markersGroup);
  }, [threshold, showHeatmap]);

  const updateMapElements = (L: any, map: any, markersGroup: any) => {
    markersGroup.clearLayers();

    if (showHeatmap) {
      filteredData.forEach(pt => {
        if (pt.probability >= threshold) {
          L.circle([pt.lat, pt.lon], {
            radius: 120,
            fillColor: '#ef4444',
            fillOpacity: pt.probability * 0.45,
            stroke: false
          }).addTo(markersGroup);
        }
      });
    } else {
      filteredData.forEach(pt => {
        const isVisible = pt.probability >= threshold;
        if (isVisible) {
          const color = interpolateColor('#3b82f6', '#ef4444', pt.probability);
          
          const marker = L.circleMarker([pt.lat, pt.lon], {
            radius: 7,
            fillColor: color,
            fillOpacity: 0.9,
            color: '#ffffff',
            weight: 1.2
          });

          marker.on('click', () => {
            setSelectedPoint(pt);
            setChatMessages(prev => [
              ...prev,
              { sender: 'ai', text: `📍 Activated Context: Location ID ${pt.id} (Pond Prob: ${(pt.probability*100).toFixed(0)}%). You can now ask: "Why did you classify this coordinate?" or click the suggestion chips below.` }
            ]);
          });

          marker.bindTooltip(`ID: ${pt.id} (${(pt.probability * 100).toFixed(0)}%)`, {
            direction: 'top',
            offset: [0, -5],
            opacity: 0.85
          });

          marker.addTo(markersGroup);
        }
      });
    }
  };

  // Local rule-based fallback expert mode
  const generateLocalAIResponse = (query: string, point: PredictionPoint | null) => {
    const q = query.toLowerCase();
    
    let contextStr = "";
    if (point) {
      contextStr = `Regarding Location ID: ${point.id} (Lat: ${point.lat.toFixed(4)}, Lon: ${point.lon.toFixed(4)}), predicted as ${point.probability >= 0.5 ? 'Aquaculture Pond' : 'Other Land Cover'} with ${(point.probability * 100).toFixed(1)}% confidence. `;
    }

    if (q.includes('leakage') || q.includes('sorting')) {
      return `${contextStr}Based on Section 4 of the FAO/ITU Trustworthiness Dossier, the Zindi dataset had target sorting leakage. We solved this by shuffling the files and evaluating exclusively using spatial GroupKFold (KMeans coordinate clustering) rather than row indices.`;
    }
    
    if (q.includes('dossier') || q.includes('trustworthiness') || q.includes('fao') || q.includes('itu')) {
      return `The FAO/ITU Trustworthiness Dossier outlines:
1. **Explainability**: Quantified SHAP value distributions and MNDWI feature partial dependence ranges.
2. **Fairness**: Regional AUC comparison between pilots (Region A vs Region B) showing equal performance (<0.05 AUC diff).
3. **Robustness**: Evaluating model decay under missing observation months and simulated optical/radar noise.
4. **Ethics**: Guidelines prohibiting automated fines and restricting surveillance.`;
    }
    
    if (q.includes('feature') || q.includes('mndwi') || q.includes('ndwi') || q.includes('radar') || q.includes('sar')) {
      return `${contextStr}Aquaculture ponds show high and stable MNDWI/NDWI water index timelines compared to seasonal croplands. Specular reflection on flat waters yields extremely low radar backscattering (Sentinel-1 VH/VV), making SAR ranges key model drivers.`;
    }

    if (point && (q.includes('point') || q.includes('location') || q.includes('why') || q.includes('explain'))) {
      return `${contextStr}Point analysis:
- **NDWI (Water index) mean**: ${point.ndwi.length ? (point.ndwi.reduce((a,b)=>a+b,0)/point.ndwi.length).toFixed(3) : 'N/A'}.
- **Radar VH mean**: ${point.vh.length ? (point.vh.reduce((a,b)=>a+b,0)/point.vh.length).toFixed(1) : 'N/A'} dB.
- **SHAP summary**: ${point.explanation}
These physical metrics support classifying this location as ${point.probability >= 0.5 ? 'a permanent water body (aquaculture pond).' : 'dryland/seasonal agriculture.'}`;
    }

    return `I am your AI Geospatial Assistant. Tap one of the suggest chips below or ask me about model features.`;
  };

  // Clickable Prompt Suggestion Chips handler
  const handleChipClick = (suggestion: string) => {
    setUserInput(suggestion);
  };

  // Chat message submit handler
  const handleSendMessage = async (e?: React.FormEvent, directMessage?: string) => {
    if (e) e.preventDefault();
    const messageToSend = directMessage || userInput;
    if (!messageToSend.trim()) return;

    setChatMessages(prev => [...prev, { sender: 'user', text: messageToSend }]);
    if (!directMessage) setUserInput('');
    setIsChatLoading(true);

    let aiText = "";
    const point = selectedPoint;

    // Construct prompt context
    const promptContext = point 
      ? `Selected Location Context: ID: ${point.id}, Lat: ${point.lat.toFixed(5)}, Lon: ${point.lon.toFixed(5)}, Predicted Aquaculture Probability: ${(point.probability*100).toFixed(1)}%, Local SHAP description: ${point.explanation}, monthly NDWI values: ${JSON.stringify(point.ndwi)}`
      : "No coordinate selected on the map.";

    const promptText = `You are a senior FAO/ITU geospatial intelligence assistant. 
Dataset Context: Target sorting leakage handled, KMeans spatial GroupKFold validation used, stacked ensemble model (LGBM+XGB+CatBoost) used.
${promptContext}
User Query: ${messageToSend}
Generate a concise, expert, helpful response:`;

    // 1. Hugging Face Serverless API
    if (apiService === 'huggingface') {
      try {
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (apiToken) {
          headers['Authorization'] = `Bearer ${apiToken}`;
        }
        
        const response = await fetch(`https://api-inference.huggingface.co/models/${selectedModel}`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            inputs: promptText,
            parameters: {
              return_full_text: false,
              max_new_tokens: 300,
              temperature: 0.7
            }
          })
        });

        if (response.ok) {
          const json = await response.json();
          // Hugging Face returns an array or object
          if (Array.isArray(json) && json[0]?.generated_text) {
            aiText = json[0].generated_text;
          } else if (json.generated_text) {
            aiText = json.generated_text;
          } else {
            aiText = JSON.stringify(json);
          }
        } else {
          const errText = await response.text();
          console.warn("Hugging Face API failed, falling back to Local Expert. Error:", errText);
          aiText = generateLocalAIResponse(messageToSend, point);
        }
      } catch (err) {
        console.error("HF Inference API request error", err);
        aiText = generateLocalAIResponse(messageToSend, point);
      }
    }
    // 2. Cloudflare Workers AI
    else if (apiService === 'cloudflare') {
      if (!cloudflareAccountId || !apiToken) {
        aiText = "Cloudflare Workers AI requires both Account ID and API Token configured in the LLM settings below.";
      } else {
        try {
          const response = await fetch(`https://api.cloudflare.com/client/v4/accounts/${cloudflareAccountId}/ai/run/${selectedModel}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiToken}`
            },
            body: JSON.stringify({
              messages: [
                { role: "system", content: "You are a senior FAO/ITU geospatial AI assistant." },
                { role: "user", content: promptText }
              ]
            })
          });

          if (response.ok) {
            const json = await response.json();
            if (json.success && json.result?.response) {
              aiText = json.result.response;
            } else {
              aiText = JSON.stringify(json.errors || json);
            }
          } else {
            aiText = `Cloudflare AI API returned error code ${response.status}. Please check details.`;
          }
        } catch (err) {
          console.error("Cloudflare AI request error", err);
          aiText = generateLocalAIResponse(messageToSend, point);
        }
      }
    }
    // 3. Local Ollama API
    else if (apiService === 'ollama') {
      try {
        const response = await fetch(`${ollamaEndpoint}/api/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: selectedModel,
            prompt: promptText,
            stream: false
          })
        });

        if (response.ok) {
          const json = await response.json();
          aiText = json.response;
        } else {
          aiText = generateLocalAIResponse(messageToSend, point);
        }
      } catch (err) {
        console.warn("Ollama unavailable, using Local Expert Mode", err);
        aiText = generateLocalAIResponse(messageToSend, point);
      }
    }
    // 4. Pure Local Expert mode
    else {
      aiText = generateLocalAIResponse(messageToSend, point);
    }

    // Clean up response if it echoes prompt parts
    if (aiText.includes("User Query:")) {
      aiText = aiText.split("User Query:")[0].trim();
    }

    setChatMessages(prev => [...prev, { sender: 'ai', text: aiText }]);
    setIsChatLoading(false);
  };

  // Hex color interpolator helper
  const interpolateColor = (color1: string, color2: string, factor: number) => {
    const c1 = parseInt(color1.slice(1), 16);
    const c2 = parseInt(color2.slice(1), 16);

    const r1 = (c1 >> 16) & 255;
    const g1 = (c1 >> 8) & 255;
    const b1 = c1 & 255;

    const r2 = (c2 >> 16) & 255;
    const g2 = (c2 >> 8) & 255;
    const b2 = c2 & 255;

    const r = Math.round(r1 + factor * (r2 - r1));
    const g = Math.round(g1 + factor * (g2 - g1));
    const b = Math.round(b1 + factor * (b2 - b1));

    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
  };

  // Sparkline SVG renderer
  const renderSparkline = (values: number[], label: string, color: string) => {
    if (!values || values.length === 0) return <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>No data</div>;

    const width = 320;
    const height = 60;
    const padding = 5;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    const points = values.map((val, idx) => {
      const x = padding + (idx / (values.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((val - min) / range) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');

    return (
      <div className="chart-container">
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
          <span className="chart-title" style={{ color }}>{label}</span>
          <span style={{ color: '#94a3b8' }}>Min: {min.toFixed(2)} | Max: {max.toFixed(2)}</span>
        </div>
        <svg width={width} height={height} className="sparkline-svg">
          <polyline
            fill="none"
            stroke={color}
            strokeWidth="2"
            points={points}
          />
          {values.map((val, idx) => {
            const x = padding + (idx / (values.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((val - min) / range) * (height - 2 * padding);
            return (
              <circle
                key={idx}
                cx={x}
                cy={y}
                r="3"
                fill={color}
                stroke="#0f172a"
                strokeWidth="1"
              />
            );
          })}
        </svg>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-title">
          <h1>FAO/ITU GeoAI Aquaculture Pond Tracker</h1>
          <p className="header-subtitle">Freshwater Controlled Environments Observation Interface & Trustworthiness Evaluator</p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button className={`btn ${mapType === 'hybrid' ? '' : 'btn-secondary'}`} onClick={() => setMapType('hybrid')}>Satellite</button>
          <button className={`btn ${mapType === 'dark' ? '' : 'btn-secondary'}`} onClick={() => setMapType('dark')}>Dark Map</button>
        </div>
      </header>

      {/* Main Container */}
      <div className="dashboard-container">
        {/* Sidebar */}
        <aside className="sidebar">
          {/* Stats Card */}
          <div className="card">
            <h3 className="card-title">Observation Summary</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-val">{stats.total}</span>
                <span className="stat-lbl">Points Checked</span>
              </div>
              <div className="stat-item">
                <span className="stat-val success">{stats.ponds}</span>
                <span className="stat-lbl">Ponds Flagged</span>
              </div>
            </div>
            <div style={{ marginTop: '0.8rem', fontSize: '0.8rem', color: '#94a3b8' }}>
              Average pond probability in selection: <strong>{(stats.avgProb * 100).toFixed(1)}%</strong>
            </div>
          </div>

          {/* Filter Card */}
          <div className="card">
            <h3 className="card-title">Detection Threshold</h3>
            <div className="slider-container">
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
              <div className="slider-labels">
                <span>0% (All Water)</span>
                <span style={{ color: '#60a5fa', fontWeight: 'bold' }}>{(threshold * 100).toFixed(0)}%</span>
                <span>100% (High Confidence)</span>
              </div>
            </div>
          </div>

          {/* Layers Card */}
          <div className="card">
            <h3 className="card-title">Layers & Filters</h3>
            <div className="control-group">
              <div className="switch-container">
                <span className="switch-label">Train Data Locations</span>
                <input
                  type="checkbox"
                  checked={showTrain}
                  onChange={(e) => setShowTrain(e.target.checked)}
                />
              </div>
              <div className="switch-container">
                <span className="switch-label">Test Data Locations</span>
                <input
                  type="checkbox"
                  checked={showTest}
                  onChange={(e) => setShowTest(e.target.checked)}
                />
              </div>
              <div className="switch-container">
                <span className="switch-label">Africa Extrapolations</span>
                <input
                  type="checkbox"
                  checked={showAfrica}
                  onChange={(e) => setShowAfrica(e.target.checked)}
                />
              </div>
              <div className="switch-container">
                <span className="switch-label">Density Heatmap Layer</span>
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                />
              </div>
              <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.6rem', marginBottom: '0.4rem' }}>
                <button
                  type="button"
                  onClick={flyToVietnam}
                  style={{
                    flex: 1,
                    backgroundColor: '#1e293b',
                    color: '#f8fafc',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    padding: '0.35rem',
                    fontSize: '0.75rem',
                    cursor: 'pointer',
                    fontWeight: '500'
                  }}
                >
                  🇻🇳 Mekong Delta
                </button>
                <button
                  type="button"
                  onClick={flyToAfrica}
                  style={{
                    flex: 1,
                    backgroundColor: '#1e293b',
                    color: '#f8fafc',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    padding: '0.35rem',
                    fontSize: '0.75rem',
                    cursor: 'pointer',
                    fontWeight: '500'
                  }}
                >
                  🌍 East Africa
                </button>
              </div>
            </div>
          </div>

          {/* AI Geospatial Consultant Card */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 className="card-title" style={{ marginBottom: 0 }}>AI Consultant</h3>
              <span style={{
                fontSize: '0.7rem',
                padding: '0.15rem 0.45rem',
                borderRadius: '9999px',
                backgroundColor: apiService === 'local' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                color: apiService === 'local' ? '#f59e0b' : '#60a5fa',
                fontWeight: 'bold',
                textTransform: 'uppercase'
              }}>
                {apiService}
              </span>
            </div>

            {/* Chat Messages Log */}
            <div style={{
              height: '180px',
              overflowY: 'auto',
              backgroundColor: '#0f172a',
              borderRadius: '8px',
              padding: '0.6rem',
              border: '1px solid #334155',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.6rem',
              fontSize: '0.8rem'
            }}>
              {chatMessages.map((msg, idx) => (
                <div key={idx} style={{
                  alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  backgroundColor: msg.sender === 'user' ? '#3b82f6' : '#1e293b',
                  color: '#f8fafc',
                  padding: '0.4rem 0.6rem',
                  borderRadius: '8px',
                  maxWidth: '85%',
                  lineHeight: '1.3',
                  wordBreak: 'break-word'
                }}>
                  {msg.text}
                </div>
              ))}
              {isChatLoading && (
                <div style={{ alignSelf: 'flex-start', color: '#94a3b8', fontSize: '0.75rem', fontStyle: 'italic' }}>
                  AI is thinking...
                </div>
              )}
            </div>

            {/* Intuitive Suggestion Chips */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
              <button
                type="button"
                onClick={() => handleChipClick("Explain this coordinate")}
                disabled={!selectedPoint}
                style={{
                  padding: '0.25rem 0.5rem',
                  borderRadius: '9999px',
                  border: '1px solid #3b82f6',
                  backgroundColor: selectedPoint ? 'rgba(59, 130, 246, 0.1)' : 'rgba(51, 65, 85, 0.2)',
                  color: selectedPoint ? '#60a5fa' : '#64748b',
                  cursor: selectedPoint ? 'pointer' : 'not-allowed',
                  fontSize: '0.7rem'
                }}
              >
                🔍 Explain Marker
              </button>
              <button
                type="button"
                onClick={() => handleChipClick("How did you handle target leakage?")}
                style={{
                  padding: '0.25rem 0.5rem',
                  borderRadius: '9999px',
                  border: '1px solid #10b981',
                  backgroundColor: 'rgba(16, 185, 129, 0.1)',
                  color: '#34d399',
                  cursor: 'pointer',
                  fontSize: '0.7rem'
                }}
              >
                🛡️ Target Leakage
              </button>
              <button
                type="button"
                onClick={() => handleChipClick("Summarize the FAO Trustworthiness Dossier")}
                style={{
                  padding: '0.25rem 0.5rem',
                  borderRadius: '9999px',
                  border: '1px solid #a855f7',
                  backgroundColor: 'rgba(168, 85, 247, 0.1)',
                  color: '#c084fc',
                  cursor: 'pointer',
                  fontSize: '0.7rem'
                }}
              >
                📊 Trustworthiness Dossier
              </button>
            </div>

            {/* Chat Input Form */}
            <form onSubmit={(e) => handleSendMessage(e)} style={{ display: 'flex', gap: '0.4rem' }}>
              <input
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder="Ask or click suggestions above..."
                disabled={isChatLoading}
                style={{
                  flex: 1,
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '0.4rem 0.6rem',
                  color: '#f8fafc',
                  fontSize: '0.8rem',
                  outline: 'none'
                }}
              />
              <button type="submit" disabled={isChatLoading} className="btn" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                Ask
              </button>
            </form>

            {/* LLM Settings Toggle */}
            <button
              onClick={() => setShowAdvancedChat(!showAdvancedChat)}
              style={{
                background: 'none',
                border: 'none',
                color: '#60a5fa',
                cursor: 'pointer',
                fontSize: '0.75rem',
                textAlign: 'left',
                padding: 0,
                outline: 'none'
              }}
            >
              {showAdvancedChat ? 'Hide LLM Settings ▴' : 'Configure LLM API Settings ▾'}
            </button>

            {/* Advanced Chat Settings */}
            {showAdvancedChat && (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.4rem',
                backgroundColor: 'rgba(0, 0, 0, 0.2)',
                padding: '0.5rem',
                borderRadius: '6px',
                border: '1px solid #334155',
                fontSize: '0.75rem'
              }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                  <label style={{ color: '#94a3b8' }}>API Service</label>
                  <select
                    value={apiService}
                    onChange={(e) => setApiService(e.target.value)}
                    style={{
                      backgroundColor: '#0f172a',
                      border: '1px solid #334155',
                      borderRadius: '4px',
                      padding: '0.2rem 0.4rem',
                      color: '#f8fafc',
                      fontSize: '0.75rem'
                    }}
                  >
                    <option value="huggingface">Hugging Face (Free API)</option>
                    <option value="cloudflare">Cloudflare Workers AI</option>
                    <option value="ollama">Local Ollama</option>
                    <option value="local">Local Expert Fallback Mode Only</option>
                  </select>
                </div>

                {apiService !== 'local' && (
                  <>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                      <label style={{ color: '#94a3b8' }}>
                        {apiService === 'ollama' ? 'Ollama Endpoint' : 'API Token / Key'}
                      </label>
                      <input
                        type={apiService === 'ollama' ? 'text' : 'password'}
                        value={apiService === 'ollama' ? ollamaEndpoint : apiToken}
                        onChange={(e) => {
                          if (apiService === 'ollama') setOllamaEndpoint(e.target.value);
                          else setApiToken(e.target.value);
                        }}
                        placeholder={apiService === 'huggingface' ? 'Paste hf_... token (optional)' : 'Enter API token'}
                        style={{
                          backgroundColor: '#0f172a',
                          border: '1px solid #334155',
                          borderRadius: '4px',
                          padding: '0.2rem 0.4rem',
                          color: '#f8fafc',
                          fontSize: '0.75rem'
                        }}
                      />
                    </div>

                    {apiService === 'cloudflare' && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                        <label style={{ color: '#94a3b8' }}>Cloudflare Account ID</label>
                        <input
                          type="text"
                          value={cloudflareAccountId}
                          onChange={(e) => setCloudflareAccountId(e.target.value)}
                          placeholder="Account ID"
                          style={{
                            backgroundColor: '#0f172a',
                            border: '1px solid #334155',
                            borderRadius: '4px',
                            padding: '0.2rem 0.4rem',
                            color: '#f8fafc',
                            fontSize: '0.75rem'
                          }}
                        />
                      </div>
                    )}

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                      <label style={{ color: '#94a3b8' }}>Model Name</label>
                      {apiService === 'huggingface' ? (
                        <select
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                          style={{
                            backgroundColor: '#0f172a',
                            border: '1px solid #334155',
                            borderRadius: '4px',
                            padding: '0.2rem 0.4rem',
                            color: '#f8fafc',
                            fontSize: '0.75rem'
                          }}
                        >
                          <option value="google/gemma-2-9b-it">Gemma 2 9B (google/gemma-2-9b-it)</option>
                          <option value="meta-llama/Meta-Llama-3-8B-Instruct">Llama 3 8B (meta-llama/Meta-Llama-3-8B-Instruct)</option>
                          <option value="mistralai/Mistral-7B-Instruct-v0.3">Mistral 7B (mistralai/Mistral-7B-Instruct-v0.3)</option>
                        </select>
                      ) : apiService === 'cloudflare' ? (
                        <select
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                          style={{
                            backgroundColor: '#0f172a',
                            border: '1px solid #334155',
                            borderRadius: '4px',
                            padding: '0.2rem 0.4rem',
                            color: '#f8fafc',
                            fontSize: '0.75rem'
                          }}
                        >
                          <option value="@cf/meta/llama-3-8b-instruct">Llama 3 8B (@cf/meta/llama-3-8b-instruct)</option>
                          <option value="@cf/mistral/mistral-7b-instruct-v0.2">Mistral 7B (@cf/mistral/mistral-7b-instruct-v0.2)</option>
                          <option value="@cf/google/gemma-7b-it-lora">Gemma 7B (@cf/google/gemma-7b-it-lora)</option>
                        </select>
                      ) : (
                        <input
                          type="text"
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                          style={{
                            backgroundColor: '#0f172a',
                            border: '1px solid #334155',
                            borderRadius: '4px',
                            padding: '0.2rem 0.4rem',
                            color: '#f8fafc',
                            fontSize: '0.75rem'
                          }}
                        />
                      )}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* System Info Card */}
          <div className="card" style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <div><strong>S1 Bands:</strong> VV/VH Polarizations</div>
            <div><strong>S2 Indices:</strong> NDWI, MNDWI, NDVI, AWEI</div>
            <div><strong>Models:</strong> Stacked LGBM + XGB + CatBoost</div>
            <div><strong>CV Strategy:</strong> Spatial KMeans GroupKFold</div>
            <div style={{ marginTop: '0.5rem', color: '#60a5fa', fontWeight: '500' }}>Powered by Leaflet & OpenStreetMap (No API Key required)</div>
          </div>
        </aside>

        {/* Map area */}
        <main className="map-container">
          {!mapsLoaded && (
            <div style={{
              position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              backgroundColor: '#0f172a', zIndex: 10
            }}>
              <p>Loading Interactive Maps API...</p>
            </div>
          )}

          <div ref={mapRef} style={{ width: '100%', height: '100%', outline: 'none' }} />

          {/* Details Overlay Panel */}
          {selectedPoint && (
            <div className="details-panel">
              <button className="panel-close-btn" onClick={() => setSelectedPoint(null)}>×</button>
              
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>Location Details</h3>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>ID: {selectedPoint.id} | Class: {selectedPoint.type.toUpperCase()}</span>
              </div>

              {/* Probability score */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: '#0f172a', padding: '0.8rem', borderRadius: '8px' }}>
                <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Pond Probability:</span>
                <span className={`probability-badge ${selectedPoint.probability >= 0.5 ? 'badge-pond' : 'badge-other'}`}>
                  {(selectedPoint.probability * 100).toFixed(1)}%
                </span>
              </div>

              {/* Rationale explanation */}
              <div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.4rem' }}>Decision Rationale (Local SHAP)</h4>
                <p style={{ fontSize: '0.85rem', lineHeight: '1.4', backgroundColor: 'rgba(59, 130, 246, 0.05)', borderLeft: '3px solid #3b82f6', padding: '0.6rem 0.8rem', borderRadius: '0 4px 4px 0' }}>
                  {selectedPoint.explanation}
                </p>
              </div>

              {/* Monthly Timelines */}
              <div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.5rem' }}>12-Month Observation Timelines</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                  {renderSparkline(selectedPoint.ndwi, 'Sentinel-2 NDWI (Water Index)', '#10b981')}
                  {renderSparkline(selectedPoint.mndwi, 'Sentinel-2 MNDWI (Modified Water Index)', '#3b82f6')}
                  {renderSparkline(selectedPoint.vh, 'Sentinel-1 SAR VH Backscatter', '#f59e0b')}
                  {renderSparkline(selectedPoint.vv, 'Sentinel-1 SAR VV Backscatter', '#a855f7')}
                </div>
              </div>

              {/* Coordinates info */}
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', borderTop: '1px solid #334155', paddingTop: '0.8rem' }}>
                <span>Lat: {selectedPoint.lat.toFixed(6)}</span>
                <span>Lon: {selectedPoint.lon.toFixed(6)}</span>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
