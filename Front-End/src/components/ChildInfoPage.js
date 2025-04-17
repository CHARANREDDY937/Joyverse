// joyverse/src/components/ChildInfoPage.js
import React from 'react';
import { useChildContext } from '../context/ChildContext';
import { useNavigate } from 'react-router-dom'; // 👈 import navigation hook
import './ChildInfoPage.css';

const ChildInfoPage = () => {
  const { childData } = useChildContext();
  const navigate = useNavigate(); // 👈 create navigate function

  const handleBrowseChildren = () => {
    navigate('/childlist'); // 👈 navigate to child list page
  };

  return (
    <div className="child-info-page">
      {childData ? (
        <div className="child-info-container">
          <h1>Therapist Portal</h1>
          <div className="child-details">
            <p><strong>Logged in as:</strong> {childData.fullName}</p>
            <p><strong>Email:</strong> {childData.email}</p>
            <button
              className="view-progress-button"
              onClick={handleBrowseChildren} // 👈 add click handler
            >
              Browse Children
            </button>
          </div>
        </div>
      ) : (
        <p>No Therapist data available.</p>
      )}
    </div>
  );
};

export default ChildInfoPage;
