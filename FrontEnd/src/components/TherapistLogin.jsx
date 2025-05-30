// // TherapistAuth.jsx
// import React, { useState, useContext } from 'react';
// import { useNavigate } from 'react-router-dom';
// import { useChildContext } from '../context/ChildContext';
// import './TherapistLogin.css'; // Assuming you named your CSS file this

// const TherapistLogin = () => {
//   const [isLogin, setIsLogin] = useState(true);
//   const [registrationSuccess, setRegistrationSuccess] = useState(false);
//   const navigate = useNavigate();
//   const { setUser } = useChildContext();
//   const [formData, setFormData] = useState({
//     username: '',
//     password: '',
//     fullName: '',
//     email: '',
//     degree: '',
//     specialization: '',
//     certificate: null,
//   });

//   const handleChange = (e) => {
//     const { name, value, files } = e.target;
//     setFormData((prev) => ({
//       ...prev,
//       [name]: files ? files[0] : value,
//     }));
//   };

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     const formPayload = new FormData();
//     for (let key in formData) {
//       if (formData[key]) {
//         formPayload.append(key, formData[key]);
//       }
//     }

//     fetch(`http://localhost:5000/therapist/${isLogin ? 'login' : 'register'}`, {
//       method: 'POST',
//       body: formPayload
//     })
//       .then(async (res) => {
//         const contentType = res.headers.get("content-type");
//         if (contentType && contentType.indexOf("application/json") !== -1) {
//           const data = await res.json();
//           if (!res.ok) {
//             throw new Error(data.message || `HTTP error! status: ${res.status}`);
//           }
//           return data;
//         } else {
//           throw new Error('Server sent an invalid response');
//         }
//       })
//       .then(data => {
//         if (isLogin) {
//           if (data.isVerified) {
//             localStorage.setItem('therapistToken', data.token);
//             setUser({
//               username: formData.username,
//               role: 'therapist',
//               isAuthenticated: true
//             });
//             navigate('/childlist');
//           } else {
//             throw new Error('Your account is pending verification. Please wait for admin approval.');
//           }
//         } else {
//           setRegistrationSuccess(true);
//           setFormData({
//             username: '',
//             password: '',
//             fullName: '',
//             email: '',
//             degree: '',
//             specialization: '',
//             certificate: null,
//           });
//         }
//       })
//       .catch(error => {
//         console.error('Error:', error);
//         alert(error.message || 'An error occurred');
//       });
//   };

//   if (registrationSuccess) {
//     return (
//       <div className="therapist-login-container" data-state="success">
//         <div className="therapist-login-box">
//           <div className="therapist-header">
//             <span className="header-icon">✅</span>
//             <h1>Registration Successful!</h1>
//             <div className="verification-message">
//               <p>Thank you for registering as a therapist!</p>
//               <p>Your application is currently under review.</p>
//               <p>Please wait for admin verification before attempting to log in.</p>
//               <p>We will process your application as soon as possible.</p>
//             </div>
//             <button 
//               className="submit-button" 
//               onClick={() => {
//                 setRegistrationSuccess(false);
//                 setIsLogin(true);
//               }}
//             >
//               Return to Login
//             </button>
//           </div>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div className="therapist-login-container" data-state={isLogin ? 'login' : 'signup'}>
//       <div className="therapist-login-box">
//         <div className="therapist-header">
//           <span className="header-icon">🧠</span>
//           <h1>{isLogin ? 'Therapist Login' : 'Therapist Sign Up'}</h1>
//           <p className="subtitle">
//             {isLogin ? 'Welcome back!' : 'Join the JoyVerse therapy team'}
//           </p>
//         </div>

//         <form onSubmit={handleSubmit} className="therapist-form">
//           {!isLogin && (
//             <>
//               <div className="form-group">
//                 <label htmlFor="fullName">Full Name</label>
//                 <input type="text" id="fullName" name="fullName" required onChange={handleChange} />
//               </div>

//               <div className="form-group">
//                 <label htmlFor="email">Email</label>
//                 <input type="email" id="email" name="email" required onChange={handleChange} />
//               </div>

//               <div className="form-group">
//                 <label htmlFor="degree">Degree</label>
//                 <input type="text" id="degree" name="degree" required onChange={handleChange} />
//               </div>

//               <div className="form-group">
//                 <label htmlFor="specialization">Specialization</label>
//                 <input type="text" id="specialization" name="specialization" required onChange={handleChange} />
//               </div>

//               <div className="form-group">
//                 <label htmlFor="certificate">Upload Certificate</label>
//                 <input type="file" id="certificate" name="certificate" accept=".pdf,.jpg,.png" required onChange={handleChange} />
//               </div>
//             </>
//           )}

//           <div className="form-group">
//             <label htmlFor="username">Username</label>
//             <input type="text" id="username" name="username" required onChange={handleChange} />
//           </div>

//           <div className="form-group">
//             <label htmlFor="password">Password</label>
//             <input type="password" id="password" name="password" required onChange={handleChange} />
//           </div>

//           <button type="submit" className="submit-button">
//             {isLogin ? 'Login' : 'Register'}
//           </button>
//         </form>

//         <div className="toggle-form">
//           {isLogin ? "Don't have an account?" : 'Already have an account?'}
//           <button className="toggle-button" onClick={() => setIsLogin(!isLogin)}>
//             {isLogin ? 'Sign up' : 'Login'}
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default TherapistLogin;
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChildContext } from '../context/ChildContext';
import './TherapistLogin.css';

const TherapistLogin = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [registrationSuccess, setRegistrationSuccess] = useState(false);
  const navigate = useNavigate();
  const { setUser } = useChildContext();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    fullName: '',
    email: '',
    degree: '',
    specialization: '',
    certificate: null,
  });

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: files ? files[0] : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const formPayload = new FormData();
    for (let key in formData) {
      if (formData[key]) {
        formPayload.append(key, formData[key]);
      }
    }

    // Base URL determined by environment
    const isProduction = process.env.NODE_ENV === 'production';
    const URL = isProduction
      ? `https://backend-brmn.onrender.com/therapist/${isLogin ? 'login' : 'register'}`
      : `http://localhost:5000/therapist/${isLogin ? 'login' : 'register'}`;

    fetch(URL, {
      method: 'POST',
      body: formPayload,
    })
      .then(async (res) => {
        const contentType = res.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.message || `HTTP error! status: ${res.status}`);
          }
          return data;
        } else {
          throw new Error('Server sent an invalid response');
        }
      })
      .then((data) => {
        if (isLogin) {
          if (data.isVerified) {
            localStorage.setItem('therapistToken', data.token);
            setUser({
              username: formData.username,
              role: 'therapist',
              isAuthenticated: true,
            });
            navigate('/childlist');
          } else {
            throw new Error('Your account is pending verification. Please wait for admin approval.');
          }
        } else {
          setRegistrationSuccess(true);
          setFormData({
            username: '',
            password: '',
            fullName: '',
            email: '',
            degree: '',
            specialization: '',
            certificate: null,
          });
        }
      })
      .catch((error) => {
        console.error('Error:', error);
        alert(error.message || 'An error occurred');
      });
  };

  if (registrationSuccess) {
    return (
      <div className="therapist-login-container" data-state="success">
        <div className="therapist-login-box">
          <div className="therapist-header">
            <span className="header-icon">✅</span>
            <h1>Registration Successful!</h1>
            <div className="verification-message">
              <p>Thank you for registering as a therapist!</p>
              <p>Your application is currently under review.</p>
              <p>Please wait for admin verification before attempting to log in.</p>
              <p>We will process your application as soon as possible.</p>
            </div>
            <button
              type="button"
              className="submit-button"
              onClick={() => {
                setRegistrationSuccess(false);
                setIsLogin(true);
              }}
            >
              Return to Login
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="therapist-login-container" data-state={isLogin ? 'login' : 'signup'}>
      <div className="therapist-login-box">
        <div className="therapist-header">
          <span className="header-icon">🧠</span>
          <h1>{isLogin ? 'Therapist Login' : 'Therapist Sign Up'}</h1>
          <p className="subtitle">{isLogin ? 'Welcome back!' : 'Join the JoyVerse therapy team'}</p>
        </div>

        <form onSubmit={handleSubmit} className="therapist-form">
          {!isLogin && (
            <>
              <div className="form-group">
                <label htmlFor="fullName">Full Name</label>
                <input
                  type="text"
                  id="fullName"
                  name="fullName"
                  value={formData.fullName}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="degree">Degree</label>
                <input
                  type="text"
                  id="degree"
                  name="degree"
                  value={formData.degree}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="specialization">Specialization</label>
                <input
                  type="text"
                  id="specialization"
                  name="specialization"
                  value={formData.specialization}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="certificate">Upload Certificate</label>
                <input
                  type="file"
                  id="certificate"
                  name="certificate"
                  accept=".pdf,.jpg,.png"
                  onChange={handleChange}
                  required
                />
              </div>
            </>
          )}

          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
            />
          </div>

          <button type="submit" className="submit-button">
            {isLogin ? 'Login' : 'Register'}
          </button>
        </form>

        <div className="toggle-form">
          <p>{isLogin ? "Don't have an account?" : 'Already have an account?'}</p>
          <button
            type="button"
            className="toggle-button"
            onClick={() => setIsLogin(!isLogin)}
          >
            {isLogin ? 'Sign up' : 'Login'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default TherapistLogin;