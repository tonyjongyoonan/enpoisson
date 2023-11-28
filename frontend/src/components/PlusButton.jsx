import "./PlusButton.css";
import React from 'react';

function PlusButton ({url}) {
    return (
        <a href={url}>
            <button class="plus-button plus"></button>
        </a>
    )
}
export default PlusButton;