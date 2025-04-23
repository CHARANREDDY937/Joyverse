import React, { useState, useEffect } from 'react';
import './childlist.css';

const ChildList = () => {
  const [users, setUsers] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/users');
        if (!response.ok) {
          throw new Error('Failed to fetch user data');
        }
        const data = await response.json();
        setUsers(data);
      } catch (error) {
        setError(error.message);
      }
    };

    fetchUsers();
  }, []);

  return (
    <div className="container">
      <h1>User Dashboard</h1>
      {error ? (
        <p className="error-message">{error}</p>
      ) : (
        <div className="user-grid">
          {users.map(user => (
            <div className="user-card" key={user._id}>
              <i className="fas fa-user-circle user-icon"></i>
              <div className="user-info">
                <p className="name">{user.name}</p>
                <p className="username">@{user.username}</p>
              </div>
              <button className="progress-btn">
                <i className="fas fa-chart-line"></i> View Progress
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ChildList;
