import '../styles/Files.css'

const Files = ({files}) => 
    <div className="files">
        {files.map(file => <span key={file}>{file}</span>)}
    </div>

export default Files