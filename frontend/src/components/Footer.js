import React from 'react'
import GitHubIcon from '@mui/icons-material/GitHub';
import './Footer.css';

function Footer() {
  return (
    <div className="footer">
        <a href="https://github.com/tonyjongyoonan/enpoisson" target="_blank" rel="noopener noreferrer">
            <GitHubIcon sx={{ color: "white" }} />
        </a>
        <p> &copy; EN POISSON</p>
    </div>
    )
}

export default Footer