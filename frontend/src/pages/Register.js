import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Register.css';

function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');

  const handleRegister = (event) => {
    event.preventDefault();
    // Handle the registration logic here
    console.log('Registering:', username, email, password);
  };

  return (
    <div className="register-container">
      <form className="register-form" onSubmit={handleRegister}>
        <h2 className="register-heading">Register</h2>
        <div className="register-input-container">
          <input
            className="register-input"
            type="text"
            placeholder="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="register-input-container">
          <input
            className="register-input"
            type="email"
            placeholder="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="register-input-container">
          <input
            className="register-input"
            type="password"
            placeholder="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <div className="register-button-container">
          <button className="register-button" type="submit">Register</button>
        </div>
        <p className="register-login-link">
          Don't have an account? <Link to="/login" className="login-link">Log In</Link>
        </p>
      </form>
    </div>
  );
}

export default Register;
