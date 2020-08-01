import React, {Component} from 'react';
import ReactDropzoneUploader from 'react-dropzone-uploader';
import Dropzone from './components/Dropzone';
import Home from './components/Home';
import 'react-dropzone-uploader/dist/styles.css';
import StarfieldAnimation from 'react-starfield-animation';
import Axios from 'axios';
import './App.css';

let max_offset,initial_offset;

class App extends Component {
  constructor(props){
    super(props);
    this.state = {
      header: false,
      s: false,
      elevation: 15.70795,
      imgUrl: '',
      preUrl: '',
    };
    this.makeShowLogo = this.makeShowLogo.bind(this);
    this.hideLogo = this.hideLogo.bind(this);
  }
  componentDidMount(){
    this.prev = window.scrollY;
    window.addEventListener('scroll',e => this.handleScroll(e));
  }

  componentDidUpdate(){
    this.handle();
  }
  handle = () => {
    if(this.state.s){
      window.scrollBy(0,window.innerHeight);
    }
  }
  handleScroll = e => {
    const window = e.currentTarget;
    if (this.prev > window.scrollY) {
        window.scrollBy(0,-window.innerHeight);
    } else if (this.prev < window.scrollY) {
      window.scrollBy(0,window.innerHeight);
    }
    this.prev = window.scrollY;
  }
  
  makeShowLogo() {
    const { header } = this.state;
    if (!header) this.setState({ header: true });
  }

  hideLogo() {
    const { header } = this.state;
    if (header) this.setState({ header: false });
  }
  render(){
    const getUploadParams = ({ meta }) => {
      // const url="https://httpbin.org/post";
      const url = "http://34.69.240.165:5000/upload";
      const f = `${url}/${encodeURIComponent(meta.name)}`;
      // console.log(formData);
      // Axios.post('http://34.69.240.165:5000/upload', formData, {
      //   headers: {
      //     'Content-Type': 'multipart/form-data',
      //     'X-Requested-With': 'XMLHttpRequest'
      //   },
      //  }).then(response => {
      //   console.log(response);
      //   this.setState({preUrl: response.segmentedImagePath})
      //   console.log(this.state.preUrl);
      // }).catch(e => {
      //   console.log(e);
      // });
      return {url, meta: {f}};
    };
  
    const handleChangeStatus = ({ meta, file }, status) => {
      // console.log(status, meta, file);
      // console.log('fjaksld');
      if(status==='done'){
      }
    };
  
    const handle = () => {
      window.scrollBy(0,window.innerHeight);
    }
    const handleSubmit = (files, allFiles) => {
      
      (files.map(f => this.setState({imgUrl:JSON.parse(f.xhr.response).segmentedImagePath,
        s:true})));
      allFiles.forEach(f => f.remove());
        console.log(JSON.parse(files[0].xhr.response));
    };
  const { imgUrl,s,elevation } = this.state;
  return (
    <div className="App">
      <div className="heading">
          SIH 2020 NM381
      </div>
        <StarfieldAnimation
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%'
        }}
      />
      <Home makeShowLogo={this.makeShowLogo} hideLogo={this.hideLogo}/>
      
    {!s&&  <ReactDropzoneUploader
      accept='image/*'
      getUploadParams={getUploadParams}
      onChangeStatus={handleChangeStatus}
      onSubmit={handleSubmit}
    />
  }
    {s&&<img src={imgUrl} alt=""className="resultImg"/>}
    <div className="dummy"> {s&&<h1 className="elevation">Angle of Elevation : {elevation}</h1>} </div>
    </div>
  );
  }
}

export default App;
