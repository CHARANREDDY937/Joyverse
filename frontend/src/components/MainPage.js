import React from "react";
import { useNavigate } from "react-router-dom";
import "./MainPage.css";

const MainPage = () => {
  const navigate = useNavigate();

  return (
    <div className="main-container">
      <h1 className="title">Welcome to JoyVerse</h1>
      <div className="teddy-container">
        <div className="teddy-bear">
        <img src="/cloud.png" alt="Teddy Bear" className="teddy-img" />

          <div className="board" onClick={() => navigate("/child-dashboard")}>
             Child <br></br> Dashboard
          </div>
        </div>
        <div className="teddy-bear">
        <img src="/cloud.png" alt="Teddy Bear" className="teddy-img" />

          <div className="board" onClick={() => navigate("/therapist-dashboard")}>
            Therapist Dashboard
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainPage;
