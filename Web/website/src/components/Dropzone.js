import React, {Component} from 'react';
import ReactDropzoneUploader from 'react-dropzone-uploader';
import DropzoneComponent from 'react-dropzone-component';
import 'react-dropzone-uploader/dist/styles.css';
import '../App.css';
import request from 'superagent';
import Axios from 'axios';

class Dropzone extends Component {
  constructor(props){
  super(props);
  this.state={
      imgUrl: '',
    };
  }
  render(){
    const getUploadParams = ({ meta }) => {
      const url = "https://httpbin.org/post";
      const fileUrl = `${url}/${encodeURIComponent(meta.name)}`;
      Axios.post(`https://httpbin.org/post`, meta, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }).then(response => {
        // JSON responses are automatically parsed.
        this.setState({imgUrl: response.segmentedImagePath})
        console.log(this.state.imgUrl);
      }).catch(e => {
        this.errors.push(e);
      });
      return {url, meta: {fileUrl}};
    };
  
    const handleChangeStatus = ({ meta, file }, status) => {
      // console.log(status, meta, file);
      // console.log('fjaksld');
      if(status==='done'){
        console.log(meta);
      }
    };
  
    const handleSubmit = (files, allFiles) => {
      console.log(files.map(f => f.meta));
      allFiles.forEach(f => f.remove());
        // console.log(response);
    };

    const eventHandlers = {
      drop: this.callbackArray,
      addedfile: this.callback,
        success: (file, response) => {
          console.log('adsf');
            console.log(response);
          }
    };
  return (
      <ReactDropzoneUploader
      // onDrop={this.onImageDrop.bind(this)}
      // config={dropzoneConfig}
      accept='image/*'
      eventHandlers={eventHandlers}
      getUploadParams={getUploadParams}
      onChangeStatus={handleChangeStatus}
      onSubmit={handleSubmit}
    />
  );
  }
}

export default Dropzone;
