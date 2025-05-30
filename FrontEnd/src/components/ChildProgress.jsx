// import { useEffect, useState } from 'react';
// import { useParams } from 'react-router-dom';
// import axios from 'axios';
// import {
//   Chart as ChartJS,
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   BarElement,  // Add this
//   Title,
//   Tooltip,
//   Legend
// } from 'chart.js';
// import { Line } from 'react-chartjs-2';
// import { Bar } from 'react-chartjs-2';
// import './ChildProgress.css';

// // Register ChartJS components
// ChartJS.register(
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   BarElement,  // Add this
//   Title,
//   Tooltip,
//   Legend
// );

// const ChildProgress = () => {
//   const { username } = useParams();
//   const [data, setData] = useState(null);
//   const [error, setError] = useState(null);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     setLoading(true);
//     console.log('Current username param:', username);
    
//     if (!username) {
//       setError('No username provided');
//       setLoading(false);
//       return;
//     }

//     // Fix the API endpoint URL
//     axios.get(`https://backend-brmn.onrender.com/api/users/progress/${username}`)
//       .then(res => {
//         console.log('Response received:', res.data);
//         setData(res.data);
//         setLoading(false);
//       })
//       .catch(err => {
//         console.error('Error details:', err.response?.data || err.message);
//         setError(err.response?.data?.message || err.message);
//         setLoading(false);
//       });
//   }, [username]);

//   if (loading) return <div className="p-6">Loading progress data...</div>;
//   if (error) return <div className="p-6 text-red-500">Error: {error}</div>;
//   if (!data || !data.emotionSummary) return <div className="p-6">No progress data available</div>;

//   const emotions = ['Happiness', 'Sadness', 'Fear', 'Disgust', 'Surprise', 'Neutral'];

//   // Remove this chartData object since we're using emotionChartData instead
//   // const chartData = { ... };

//   const emotionChartData = {
//     labels: data.emotionSummary.map((e, i) => `Session ${i + 1}`),
//     datasets: emotions.map((emotion, idx) => ({
//       label: emotion,
//       data: data.emotionSummary.map(e => e.emotion[emotion]),
//       borderColor: `hsl(${idx * 60}, 70%, 50%)`,
//       tension: 0.4,
//       fill: false,
//     }))
//   };

//   const gameChartData = {
//     labels: data.emotionSummary.map(e => e.game || 'Unknown Game'),
//     datasets: [{
//       label: 'Game Performance',
//       data: data.emotionSummary.map(e => 
//         Object.values(e.emotion).reduce((a, b) => a + b, 0) / Object.values(e.emotion).length
//       ),
//       backgroundColor: 'rgba(109, 40, 217, 0.8)',
//       borderRadius: 6,
//     }]
//   };

//   return (
//     <div className="progress-container">
//       <h2 className="progress-title">{data.name}'s Progress</h2>

//       <div className="charts-grid">
//         <div className="chart-container">
//           <h3 className="chart-title">Emotion Recognition Progress</h3>
//           <Line data={emotionChartData} options={{
//             responsive: true,
//             scales: {
//               y: {
//                 beginAtZero: true,
//                 max: 100,
//                 grid: { color: 'rgba(0, 0, 0, 0.1)' }
//               },
//               x: { grid: { display: false } }
//             }
//           }} />
//         </div>

//         <div className="chart-container">
//           <h3 className="chart-title">Game Performance</h3>
//           <Bar data={gameChartData} options={{
//             responsive: true,
//             scales: {
//               y: {
//                 beginAtZero: true,
//                 max: 100,
//                 grid: { color: 'rgba(0, 0, 0, 0.1)' }
//               },
//               x: { grid: { display: false } }
//             }
//           }} />
//         </div>
//       </div>

//       <div className="sessions-container">
//         <h3 className="chart-title">Session Details</h3>
//         {data.emotionSummary.map((session, idx) => (
//           <div key={idx} className="session-card">
//             <p><strong>Game:</strong> {session.game}</p>
//             <p><strong>Timestamp:</strong> {new Date(session.timestamp).toLocaleString()}</p>
//             <ul className="session-details">
//               {Object.entries(session.emotion).map(([emo, val]) => (
//                 <li key={emo}>{emo}: {val}%</li>
//               ))}
//             </ul>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default ChildProgress;
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import './ChildProgress.css';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ChildProgress = () => {
  const { username } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    console.log('Current username param:', username);
    
    if (!username) {
      setError('No username provided');
      setLoading(false);
      return;
    }

    // Base URL determined by environment
    const isProduction = process.env.NODE_ENV === 'production';
    const URL = isProduction
      ? `https://backend-brmn.onrender.com/api/users/progress/${username}`
      : `http://localhost:5000/api/users/progress/${username}`;

    axios.get(URL)
      .then(res => {
        console.log('Response received:', res.data);
        setData(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error details:', err.response?.data || err.message);
        setError(err.response?.data?.message || err.message);
        setLoading(false);
      });
  }, [username]);

  if (loading) return <div className="p-6">Loading progress data...</div>;
  if (error) return <div className="p-6 text-red-500">Error: {error}</div>;
  if (!data || !data.emotionSummary) return <div className="p-6">No progress data available</div>;

  const emotions = ['Happiness', 'Sadness', 'Fear', 'Disgust', 'Surprise', 'Neutral'];

  const emotionChartData = {
    labels: data.emotionSummary.map((e, i) => `Session ${i + 1}`),
    datasets: emotions.map((emotion, idx) => ({
      label: emotion,
      data: data.emotionSummary.map(e => e.emotion[emotion]),
      borderColor: `hsl(${idx * 60}, 70%, 50%)`,
      tension: 0.4,
      fill: false,
    }))
  };

  const gameChartData = {
    labels: data.emotionSummary.map(e => e.game || 'Unknown Game'),
    datasets: [{
      label: 'Game Performance',
      data: data.emotionSummary.map(e => 
        Object.values(e.emotion).reduce((a, b) => a + b, 0) / Object.values(e.emotion).length
      ),
      backgroundColor: 'rgba(109, 40, 217, 0.8)',
      borderRadius: 6,
    }]
  };

  return (
    <div className="progress-container">
      <h2 className="progress-title">{data.name}'s Progress</h2>

      <div className="charts-grid">
        <div className="chart-container">
          <h3 className="chart-title">Emotion Recognition Progress</h3>
          <Line data={emotionChartData} options={{
            responsive: true,
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                grid: { color: 'rgba(0, 0, 0, 0.1)' }
              },
              x: { grid: { display: false } }
            }
          }} />
        </div>

        <div className="chart-container">
          <h3 className="chart-title">Game Performance</h3>
          <Bar data={gameChartData} options={{
            responsive: true,
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                grid: { color: 'rgba(0, 0, 0, 0.1)' }
              },
              x: { grid: { display: false } }
            }
          }} />
        </div>
      </div>

      <div className="sessions-container">
        <h3 className="chart-title">Session Details</h3>
        {data.emotionSummary.map((session, idx) => (
          <div key={idx} className="session-card">
            <p><strong>Game:</strong> {session.game}</p>
            <p><strong>Timestamp:</strong> {new Date(session.timestamp).toLocaleString()}</p>
            <ul className="session-details">
              {Object.entries(session.emotion).map(([emo, val]) => (
                <li key={emo}>{emo}: {val}%</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChildProgress;