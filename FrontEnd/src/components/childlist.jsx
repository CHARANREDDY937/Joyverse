// // import React, { useState, useEffect } from 'react';
// // import { motion } from 'framer-motion';
// // import { useNavigate } from 'react-router-dom';
// // import { useChildContext } from '../context/ChildContext';
// // import './childlist.css';

// // const ChildList = () => {
// //   const navigate = useNavigate();
// //   const { setChildData, user } = useChildContext();
// //   const [users, setUsers] = useState([]);
// //   const [error, setError] = useState(null);
// //   const [loading, setLoading] = useState(true);
// //   const [searchTerm, setSearchTerm] = useState('');

// //   // Protect route
// //   useEffect(() => {
// //     if (!user || user.role !== 'therapist') {
// //       navigate('/therapist');
// //     }
// //   }, [user, navigate]);

// //   useEffect(() => {
// //     const fetchUsers = async () => {
// //       try {
// //         const response = await fetch('http://localhost:5000/api/users');
// //         if (!response.ok) {
// //           throw new Error('Failed to fetch user data');
// //         }
// //         const data = await response.json();
// //         setUsers(data);
// //       } catch (error) {
// //         setError(error.message);
// //       } finally {
// //         setLoading(false);
// //       }
// //     };

// //     fetchUsers();
// //   }, []);

// //   const filteredUsers = users.filter(user =>
// //     user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
// //     user.username.toLowerCase().includes(searchTerm.toLowerCase())
// //   );

// //   const handleViewProgress = async (user) => {
// //     try {
// //       const response = await fetch(`http://localhost:5000/api/users/progress/${user.username}`);
// //       if (!response.ok) {
// //         throw new Error('Failed to fetch progress data');
// //       }
// //       const progressData = await response.json();

// //       setChildData({
// //         username: user.username,
// //         progressData,
// //         name: user.name
// //       });

// //       navigate(`/child-progress/${user.username}`);
// //     } catch (error) {
// //       console.error('Error fetching progress:', error);
// //       setError('Failed to load child progress. Please try again.');
// //     }
// //   };

// //   if (loading) {
// //     return (
// //       <div className="loading-container">
// //         <div className="loader"></div>
// //         <p>Loading children's data...</p>
// //       </div>
// //     );
// //   }

// //   return (
// //     <motion.div 
// //       className="child-list-container"
// //       initial={{ opacity: 0 }}
// //       animate={{ opacity: 1 }}
// //       transition={{ duration: 0.5 }}
// //     >
// //       <div className="dashboard-header">
// //         <h1>Children's Progress Dashboard</h1>
// //         <div className="search-bar">
// //           <input
// //             type="text"
// //             placeholder="Search by name or username..."
// //             value={searchTerm}
// //             onChange={(e) => setSearchTerm(e.target.value)}
// //           />
// //         </div>
// //       </div>

// //       {error ? (
// //         <motion.div 
// //           className="error-message"
// //           initial={{ opacity: 0, y: -20 }}
// //           animate={{ opacity: 1, y: 0 }}
// //         >
// //           {error}
// //         </motion.div>
// //       ) : (
// //         <div className="user-grid">
// //           {filteredUsers.map((user, index) => (
// //             <motion.div
// //               key={user._id}
// //               className="user-card"
// //               initial={{ opacity: 0, y: 20 }}
// //               animate={{ opacity: 1, y: 0 }}
// //               transition={{ delay: index * 0.1 }}
// //               whileHover={{ scale: 1.02 }}
// //             >
// //               <div className="user-header">
// //                 <div className="user-avatar">
// //                   <span>👤</span>
// //                 </div>
// //                 <div className="user-info">
// //                   <h3>{user.name}</h3>
// //                   <p>@{user.username}</p>
// //                 </div>
// //               </div>

// //               {/* <div className="user-stats">
// //                 <div className="stat">
// //                   <span>🎮</span>
// //                   <span>Games: {user.gamesPlayed || 0}</span>
// //                 </div>
// //                 <div className="stat">
// //                   <span>⭐</span>
// //                   <span>Score: {user.totalScore || 0}</span>
// //                 </div>
// //               </div> */}

// //               <div className="button-group">
// //                 <motion.button
// //                   className="view-progress-btn"
// //                   onClick={() => handleViewProgress(user)}
// //                   whileHover={{ scale: 1.05 }}
// //                   whileTap={{ scale: 0.95 }}
// //                 >
// //                   <span>📈</span>
// //                   View Progress
// //                 </motion.button>
// //               </div>
// //             </motion.div>
// //           ))}
// //         </div>
// //       )}
// //     </motion.div>
// //   );
// // };

// // export default ChildList;

// import React, { useState, useEffect } from 'react';
// import { motion } from 'framer-motion';
// import { useNavigate } from 'react-router-dom';
// import { useChildContext } from '../context/ChildContext';
// import './childlist.css';

// // Determine API base URL based on environment
// const isProduction = process.env.NODE_ENV === 'production';
// const API_URL = isProduction
//   ? 'https://backend-brmn.onrender.com'
//   : 'http://localhost:5000';

// const ChildList = () => {
//   const navigate = useNavigate();
//   const { setChildData, user } = useChildContext();
//   const [users, setUsers] = useState([]);
//   const [error, setError] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [searchTerm, setSearchTerm] = useState('');

//   // Protect route
//   useEffect(() => {
//     if (!user || user.role !== 'therapist') {
//       navigate('/therapist');
//     }
//   }, [user, navigate]);

//   useEffect(() => {
//     const fetchUsers = async () => {
//       try {
//         const response = await fetch(`${API_URL}/api/users`);
//         if (!response.ok) {
//           throw new Error('Failed to fetch user data');
//         }
//         const data = await response.json();
//         setUsers(data);
//       } catch (error) {
//         setError(error.message);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchUsers();
//   }, []);

//   const filteredUsers = users.filter(user =>
//     user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
//     user.username.toLowerCase().includes(searchTerm.toLowerCase())
//   );

//   const handleViewProgress = async (user) => {
//     try {
//       const response = await fetch(`${API_URL}/api/users/progress/${user.username}`);
//       if (!response.ok) {
//         throw new Error('Failed to fetch progress data');
//       }
//       const progressData = await response.json();

//       setChildData({
//         username: user.username,
//         progressData,
//         name: user.name
//       });

//       navigate(`/child-progress/${user.username}`);
//     } catch (error) {
//       console.error('Error fetching progress:', error);
//       setError('Failed to load child progress. Please try again.');
//     }
//   };

//   if (loading) {
//     return (
//       <div className="loading-container">
//         <div className="loader"></div>
//         <p>Loading children's data...</p>
//       </div>
//     );
//   }

//   return (
//     <motion.div 
//       className="child-list-container"
//       initial={{ opacity: 0 }}
//       animate={{ opacity: 1 }}
//       transition={{ duration: 0.5 }}
//     >
//       <div className="dashboard-header">
//         <h1>Children's Progress Dashboard</h1>
//         <div className="search-bar">
//           <input
//             type="text"
//             placeholder="Search by name or username..."
//             value={searchTerm}
//             onChange={(e) => setSearchTerm(e.target.value)}
//           />
//         </div>
//       </div>

//       {error ? (
//         <motion.div 
//           className="error-message"
//           initial={{ opacity: 0, y: -20 }}
//           animate={{ opacity: 1, y: 0 }}
//         >
//           {error}
//         </motion.div>
//       ) : (
//         <div className="user-grid">
//           {filteredUsers.map((user, index) => (
//             <motion.div
//               key={user._id}
//               className="user-card"
//               initial={{ opacity: 0, y: 20 }}
//               animate={{ opacity: 1, y: 0 }}
//               transition={{ delay: index * 0.1 }}
//               whileHover={{ scale: 1.02 }}
//             >
//               <div className="user-header">
//                 <div className="user-avatar">
//                   <span>👤</span>
//                 </div>
//                 <div className="user-info">
//                   <h3>{user.name}</h3>
//                   <p>@{user.username}</p>
//                 </div>
//               </div>

//               {/* <div className="user-stats">
//                 <div className="stat">
//                   <span>🎮</span>
//                   <span>Games: {user.gamesPlayed || 0}</span>
//                 </div>
//                 <div className="stat">
//                   <span>⭐</span>
//                   <span>Score: {user.totalScore || 0}</span>
//                 </div>
//               </div> */}

//               <div className="button-group">
//                 <motion.button
//                   className="view-progress-btn"
//                   onClick={() => handleViewProgress(user)}
//                   whileHover={{ scale: 1.05 }}
//                   whileTap={{ scale: 0.95 }}
//                 >
//                   <span>📈</span>
//                   View Progress
//                 </motion.button>
//               </div>
//             </motion.div>
//           ))}
//         </div>
//       )}
//     </motion.div>
//   );
// };

// export default ChildList;
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useChildContext } from '../context/ChildContext';
import './childlist.css';

// Use Render backend URL exclusively
const API_URL = 'https://backend-brmn.onrender.com';

const ChildList = () => {
  const navigate = useNavigate();
  const { setChildData, user } = useChildContext();
  const [users, setUsers] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  // Protect route
  useEffect(() => {
    if (!user || user.role !== 'therapist') {
      navigate('/therapist');
    }
  }, [user, navigate]);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch(`${API_URL}/api/users`);
        if (!response.ok) {
          throw new Error('Failed to fetch user data');
        }
        const data = await response.json();
        setUsers(data);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();
  }, []);

  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.username.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleViewProgress = async (user) => {
    try {
      const response = await fetch(`${API_URL}/api/users/progress/${user.username}`);
      if (!response.ok) {
        throw new Error('Failed to fetch progress data');
      }
      const progressData = await response.json();

      setChildData({
        username: user.username,
        progressData,
        name: user.name
      });

      navigate(`/child-progress/${user.username}`);
    } catch (error) {
      console.error('Error fetching progress:', error);
      setError('Failed to load child progress. Please try again.');
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loader"></div>
        <p>Loading children's data...</p>
      </div>
    );
  }

  return (
    <motion.div 
      className="child-list-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="dashboard-header">
        <h1>Children's Progress Dashboard</h1>
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search by name or username..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {error ? (
        <motion.div 
          className="error-message"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {error}
        </motion.div>
      ) : (
        <div className="user-grid">
          {filteredUsers.map((user, index) => (
            <motion.div
              key={user._id}
              className="user-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="user-header">
                <div className="user-avatar">
                  <span>👤</span>
                </div>
                <div className="user-info">
                  <h3>{user.name}</h3>
                  <p>@{user.username}</p>
                </div>
              </div>

              <div className="button-group">
                <motion.button
                  className="view-progress-btn"
                  onClick={() => handleViewProgress(user)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>📈</span>
                  View Progress
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default ChildList;