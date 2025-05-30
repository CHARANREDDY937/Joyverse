// import React, { useState, useEffect } from 'react';
// import './AdminDashboard.css';

// const AdminDashboard = () => {
//   const [therapists, setTherapists] = useState([]);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     fetchTherapists();
//   }, []);

//   const fetchTherapists = async () => {
//     try {
//       const adminToken = localStorage.getItem('adminToken');
//       if (!adminToken) {
//         setError('Not authenticated');
//         // Optionally redirect to login
//         return;
//       }

//       const response = await fetch('http://localhost:5000/admin/therapists', {
//         headers: {
//           'Authorization': `Bearer ${adminToken}`, // Added backticks here
//           'Content-Type': 'application/json'
//         }
//       });

//       if (response.status === 401) {
//         setError('Session expired. Please login again');
//         localStorage.removeItem('adminToken');
//         // Optionally redirect to login
//         return;
//       }

//       if (!response.ok) {
//         throw new Error('Failed to fetch therapists');
//       }

//       const data = await response.json();
//       setTherapists(data);
//       setError(null); // Clear any previous errors
//     } catch (err) {
//       console.error('Error fetching therapists:', err);
//       setError('Failed to load therapists');
//     }
//   };

//   const handleVerification = async (therapistId, isVerified) => {
//     try {
//       const response = await fetch(`http://localhost:5000/admin/verify-therapist/${therapistId}`, {
//         method: 'PUT',
//         headers: {
//           'Authorization': `Bearer ${localStorage.getItem('adminToken')}`, // Added backticks here
//           'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({ isVerified })
//       });

//       if (!response.ok) {
//         throw new Error('Failed to update verification status');
//       }

//       const data = await response.json();
//       if (data.success) {
//         alert(data.message); // Using the message from response
//         fetchTherapists();
//       } else {
//         throw new Error(data.message || 'Verification failed');
//       }
//     } catch (err) {
//       console.error('Error updating verification:', err);
//       alert(err.message || 'Failed to update verification status. Please try again.');
//     }
//   };

//   if (error) {
//     return <div className="error-message">{error}</div>;
//   }

//   return (
//     <div className="admin-dashboard">
//       <h1>Admin Dashboard</h1>
//       <h2>Therapist Verification</h2>
//       <div className="therapists-list">
//         {therapists.length === 0 ? (
//           <p>No therapists found</p>
//         ) : (
//           therapists.map(therapist => (
//             <div key={therapist._id} className="therapist-card">
//               <h3>{therapist.fullName}</h3>
//               <p>Username: {therapist.username}</p>
//               <p>Email: {therapist.email}</p>
//               <p>Degree: {therapist.degree}</p>
//               <p>Specialization: {therapist.specialization}</p>
//               {therapist.certificatePath && (
//                 <a 
//                   href={`http://localhost:5000/${therapist.certificatePath}`} 
//                   target="_blank" 
//                   rel="noopener noreferrer"
//                 >
//                   View Certificate
//                 </a>
//               )}
//               <div className="verification-controls">
//                 <button 
//                   onClick={() => handleVerification(therapist._id, true)}
//                   className={`verify-btn ${therapist.isVerified ? 'active' : ''}`}
//                 >
//                   Verify
//                 </button>
//                 <button 
//                   onClick={() => handleVerification(therapist._id, false)}
//                   className={`reject-btn ${!therapist.isVerified ? 'active' : ''}`}
//                 >
//                   Reject
//                 </button>
//               </div>
//             </div>
//           ))
//         )}
//       </div>
//     </div>
//   );
// };

// export default AdminDashboard;
import React, { useState, useEffect } from 'react';
import './AdminDashboard.css';

const AdminDashboard = () => {
  const [therapists, setTherapists] = useState([]);
  const [error, setError] = useState(null);

  // Base URL for API calls, determined by environment
  const isProduction = process.env.NODE_ENV === 'production';
  const BASE_URL = isProduction
    ? 'https://backend-brmn.onrender.com'
    : 'http://localhost:5000';

  useEffect(() => {
    fetchTherapists();
  }, []);

  const fetchTherapists = async () => {
    try {
      const adminToken = localStorage.getItem('adminToken');
      if (!adminToken) {
        setError('Not authenticated');
        // Optionally redirect to login
        return;
      }

      const response = await fetch(`${BASE_URL}/admin/therapists`, {
        headers: {
          Authorization: `Bearer ${adminToken}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.status === 401) {
        setError('Session expired. Please login again');
        localStorage.removeItem('adminToken');
        // Optionally redirect to login
        return;
      }

      if (!response.ok) {
        throw new Error('Failed to fetch therapists');
      }

      const data = await response.json();
      setTherapists(data);
      setError(null); // Clear any previous errors
    } catch (err) {
      console.error('Error fetching therapists:', err);
      setError('Failed to load therapists');
    }
  };

  const handleVerification = async (therapistId, isVerified) => {
    try {
      const response = await fetch(`${BASE_URL}/admin/verify-therapist/${therapistId}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('adminToken')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ isVerified }),
      });

      if (!response.ok) {
        throw new Error('Failed to update verification status');
      }

      const data = await response.json();
      if (data.success) {
        alert(data.message); // Using the message from response
        fetchTherapists();
      } else {
        throw new Error(data.message || 'Verification failed');
      }
    } catch (err) {
      console.error('Error updating verification:', err);
      alert(err.message || 'Failed to update verification status. Please try again.');
    }
  };

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="admin-dashboard">
      <h1>Admin Dashboard</h1>
      <h2>Therapist Verification</h2>
      <div className="therapists-list">
        {therapists.length === 0 ? (
          <p>No therapists found</p>
        ) : (
          therapists.map((therapist) => (
            <div key={therapist._id} className="therapist-card">
              <h3>{therapist.fullName}</h3>
              <p>Username: {therapist.username}</p>
              <p>Email: {therapist.email}</p>
              <p>Degree: {therapist.degree}</p>
              <p>Specialization: {therapist.specialization}</p>
              {therapist.certificatePath && (
                <a
                  href={`${BASE_URL}/${therapist.certificatePath}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View Certificate
                </a>
              )}
              <div className="verification-controls">
                <button
                  onClick={() => handleVerification(therapist._id, true)}
                  className={`verify-btn ${therapist.isVerified ? 'active' : ''}`}
                >
                  Verify
                </button>
                <button
                  onClick={() => handleVerification(therapist._id, false)}
                  className={`reject-btn ${!therapist.isVerified ? 'active' : ''}`}
                >
                  Reject
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AdminDashboard;