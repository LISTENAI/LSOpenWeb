"use strict";(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[7705],{3905:function(e,t,n){n.d(t,{Zo:function(){return s},kt:function(){return m}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var p=r.createContext({}),u=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},s=function(e){var t=u(e.components);return r.createElement(p.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,p=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),d=u(n),m=i,f=d["".concat(p,".").concat(m)]||d[m]||c[m]||o;return n?r.createElement(f,a(a({ref:t},s),{},{components:n})):r.createElement(f,a({ref:t},s))}));function m(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,a=new Array(o);a[0]=d;var l={};for(var p in t)hasOwnProperty.call(t,p)&&(l[p]=t[p]);l.originalType=e,l.mdxType="string"==typeof e?e:i,a[1]=l;for(var u=2;u<o;u++)a[u]=n[u];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},6905:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return p},metadata:function(){return u},toc:function(){return s},default:function(){return d}});var r=n(7462),i=n(3366),o=(n(7294),n(3905)),a=["components"],l={sidebar_position:1},p="\u5236\u4f5c\u79bb\u5728\u7ebf\u9879\u76ee\u7684CSK\u56fa\u4ef6",u={unversionedId:"AIsolution/dsp/firmware_development/CSK_online_firmware",id:"AIsolution/dsp/firmware_development/CSK_online_firmware",isDocsHomePage:!1,title:"\u5236\u4f5c\u79bb\u5728\u7ebf\u9879\u76ee\u7684CSK\u56fa\u4ef6",description:"\u672c\u6587\u4e3b\u8981\u4ecb\u7ecd\u79bb\u5728\u7ebf\u9879\u76ee CSK \u56fa\u4ef6\u7684\u5236\u4f5c\u65b9\u6cd5\uff0c\u901a\u8fc7\u9605\u8bfb\u672c\u6587\uff0c\u4f60\u5c06\u4e86\u89e3\u5230\uff1a",source:"@site/docs/AIsolution/dsp/firmware_development/CSK_online_firmware.md",sourceDirName:"AIsolution/dsp/firmware_development",slug:"/AIsolution/dsp/firmware_development/CSK_online_firmware",permalink:"/AIsolution/dsp/firmware_development/CSK_online_firmware",version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"AIsolutionDSP",previous:{title:"\u786c\u4ef6\u53c2\u8003\u8bbe\u8ba1",permalink:"/AIsolution/dsp/Hardwaredevelopment/hardware_guide"},next:{title:"\u521b\u5efa\u4f60\u7684 iFLYOS \u8bbe\u5907",permalink:"/AIsolution/dsp/firmware_development/Create_iFLYOS_equipment"}},s=[{value:"1.\u4e0b\u8f7d LSTudio",id:"1\u4e0b\u8f7d-lstudio",children:[]},{value:"2.\u5728 LSTudio \u521b\u5efa\u9879\u76ee",id:"2\u5728-lstudio-\u521b\u5efa\u9879\u76ee",children:[]},{value:"3.\u97f3\u9891\u8f93\u51fa\u914d\u7f6e",id:"3\u97f3\u9891\u8f93\u51fa\u914d\u7f6e",children:[]},{value:"4.\u4ea4\u4e92\u914d\u7f6e",id:"4\u4ea4\u4e92\u914d\u7f6e",children:[]},{value:"5.\u56fa\u4ef6\u6253\u5305\u4e0e\u70e7\u5f55",id:"5\u56fa\u4ef6\u6253\u5305\u4e0e\u70e7\u5f55",children:[]}],c={toc:s};function d(e){var t=e.components,l=(0,i.Z)(e,a);return(0,o.kt)("wrapper",(0,r.Z)({},c,l,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"\u5236\u4f5c\u79bb\u5728\u7ebf\u9879\u76ee\u7684csk\u56fa\u4ef6"},"\u5236\u4f5c\u79bb\u5728\u7ebf\u9879\u76ee\u7684CSK\u56fa\u4ef6"),(0,o.kt)("blockquote",null,(0,o.kt)("p",{parentName:"blockquote"},"\u672c\u6587\u4e3b\u8981\u4ecb\u7ecd\u79bb\u5728\u7ebf\u9879\u76ee CSK \u56fa\u4ef6\u7684\u5236\u4f5c\u65b9\u6cd5\uff0c\u901a\u8fc7\u9605\u8bfb\u672c\u6587\uff0c\u4f60\u5c06\u4e86\u89e3\u5230\uff1a"),(0,o.kt)("p",{parentName:"blockquote"},"\u5982\u4f55\u5728 LSTudio \u914d\u7f6e\u3001\u6253\u5305\u3001\u5e76\u70e7\u5f55 LSKIts \u79bb\u5728\u7ebf\u5f00\u53d1\u677f\u7684 CSK \u56fa\u4ef6\uff1b")),(0,o.kt)("h3",{id:"1\u4e0b\u8f7d-lstudio"},"1.\u4e0b\u8f7d LSTudio"),(0,o.kt)("p",null,"LSTudio \u4e3b\u8981\u7528\u4e8e CSK \u56fa\u4ef6\u914d\u7f6e\u4e0e\u70e7\u5f55\uff0c\u5728 CSK \u9879\u76ee\u5f00\u53d1\u6307\u5357\u4e2d\u7684",(0,o.kt)("a",{parentName:"p",href:"AIsolution/ESR/Quick_start/Quick_start"},"\u300a\u5feb\u901f\u5165\u95e8CSK\u9879\u76ee\u5f00\u53d1\u300b"),"\u5df2\u8be6\u7ec6\u8bf4\u660e\u5982\u4f55\u4e0b\u8f7d\u5b89\u88c5 LStudio\uff0c\u6b64\u5904\u4e0d\u518d\u505a\u8fc7\u591a\u63cf\u8ff0\u3002"),(0,o.kt)("h3",{id:"2\u5728-lstudio-\u521b\u5efa\u9879\u76ee"},"2.\u5728 LSTudio \u521b\u5efa\u9879\u76ee"),(0,o.kt)("p",null,"\u5b89\u88c5 LSTudio \u540e\uff0c\u9f20\u6807\u60ac\u505c\u81f3\u9876\u90e8\u83dc\u5355\u680f\u7684\u3010\u9879\u76ee\u3011\uff0c\u9009\u62e9\u3010\u65b0\u5efa\u3011\uff0c\u5982\u4e0b\u56fe\uff1b\u9009\u62e9\u3010\u56fa\u4ef6\u5f00\u53d1\u9879\u76ee\u3011\uff1b"),(0,o.kt)("p",null,"\u3010\u82af\u7247\u578b\u53f7\u53ca\u65b9\u6848\u3011\u9009\u62e9 ",(0,o.kt)("strong",{parentName:"p"},"4002"),"\uff1b"),(0,o.kt)("p",null,"\u3010\u57fa\u7840\u56fa\u4ef6\u7248\u672c\u3011\u9009\u62e9",(0,o.kt)("strong",{parentName:"p"},"3.0.2"),"\uff08\u76f4\u63a5\u9009\u7ebf\u4e0a\u6700\u65b0\u7684\u7248\u672c\u5373\u53ef\uff09\uff1b"),(0,o.kt)("p",null,"\u3010\u677f\u578b\u6a21\u677f\u3011\u9009\u62e9",(0,o.kt)("strong",{parentName:"p"},"\u8046\u601d\u5f00\u53d1\u677fV1.0"),"\uff1b"),(0,o.kt)("p",null,(0,o.kt)("img",{src:n(827).Z})),(0,o.kt)("h3",{id:"3\u97f3\u9891\u8f93\u51fa\u914d\u7f6e"},"3.\u97f3\u9891\u8f93\u51fa\u914d\u7f6e"),(0,o.kt)("p",null,"\u9879\u76ee\u521b\u5efa\u6210\u529f\u540e\uff0c\u627e\u5230 ",(0,o.kt)("inlineCode",{parentName:"p"},"application.lini")," \u914d\u7f6e\u6587\u4ef6\uff0c\u6587\u4ef6\u5177\u4f53\u8def\u5f84\u5982\u4e0b\uff1a ",(0,o.kt)("inlineCode",{parentName:"p"},"./config/environment/application.lini"),"\uff1b"),(0,o.kt)("p",null,"\u9996\u5148\uff0c\u4f60\u9700\u8981\u8bbe\u7f6e\u56fa\u4ef6\u7684\u534f\u8bae\u6a21\u5f0f\uff0c\u5728\u6587\u4ef6\u4e2d\u627e\u5230  ",(0,o.kt)("inlineCode",{parentName:"p"},"sys_mode")," \u5b57\u6bb5\uff0c\u5e76\u5c06\u503c\u8bbe\u7f6e\u4e3a ",(0,o.kt)("inlineCode",{parentName:"p"},"public"),"  \u3002\u5982\u4e0b\u56fe\uff1a"),(0,o.kt)("p",null,(0,o.kt)("img",{src:n(3043).Z})),(0,o.kt)("p",null,"\u4e4b\u540e\uff0c\u4f60\u9700\u8981\u4fee\u6539 CSK \u8f93\u51fa\u7684\u97f3\u9891\u901a\u9053\uff0c4002 \u56fa\u4ef6\u9ed8\u8ba4\u8f93\u51fa\u7684\u97f3\u9891\u901a\u9053\u5206\u522b\u662f ",(0,o.kt)("strong",{parentName:"p"},"[1,2,5,6]"),"   \uff0c\u5728\u79bb\u5728\u7ebf\u9879\u76ee\u4e2d\uff0c\u9700\u8981\u4fee\u6539\u4e3a  ",(0,o.kt)("strong",{parentName:"p"},"[1.2,8,3]")," \u3002CSK \u603b\u5171\u53ef\u8f93\u51fa 8 \u8def\u97f3\u9891\uff0c\u53ef\u9009\u62e9 4 \u8def\u8f93\u51fa\u7ed9\u4e0a\u4f4d\u673a\u30028 \u8def\u97f3\u9891\u5206\u522b\u7684\u542b\u4e49\u53ef\u53c2\u8003\u300aLISA API\u300b\u4e2d ",(0,o.kt)("inlineCode",{parentName:"p"},"/Modules/CSK/csk_record_set_i2s_ch")," \u7684\u5b9a\u4e49\u3002"),(0,o.kt)("p",null,(0,o.kt)("img",{src:n(2611).Z})),(0,o.kt)("h3",{id:"4\u4ea4\u4e92\u914d\u7f6e"},"4.\u4ea4\u4e92\u914d\u7f6e"),(0,o.kt)("p",null,"\u8fdb\u5165\u4ea4\u4e92\u914d\u7f6e\uff0c\u7531\u4e8e\u76ee\u524d\u79bb\u5728\u7ebf\u9879\u76ee\u4e2d\uff0c\u5c1a\u4e0d\u652f\u6301\u79bb\u7ebf\u8bc6\u522b\uff0c\u6240\u6709\u4ea4\u4e92\u90fd\u4e3a\u5728\u7ebf\u4ea4\u4e92\u3002\u4e14\u64ad\u653e\u5668\u7531\u4e0a\u4f4d\u673a\u63a7\u5236\uff0cCSK\u4e0d\u76f4\u63a5\u64ad\u653e\u97f3\u9891\uff0c\u6240\u4ee5\u6b64\u5904\u53ef\u4e0d\u8bbe\u7f6e\u79bb\u7ebf\u547d\u4ee4\u8bcd\u4ee5\u53ca\u5bf9\u5e94\u56de\u590d\u8bed\u3002\u5982\u4f55\u5728\u4e0a\u4f4d\u673a\u8bbe\u7f6e\u81ea\u5b9a\u4e49\u56de\u590d\u8bed\uff0c\u53ef\u53c2\u8003",(0,o.kt)("a",{parentName:"p",href:"/AIsolution/dsp/firmware_development/xr872_evs"},"\u300a\u4e0a\u4f4d\u673a\u56fa\u4ef6\u4e8c\u6b21\u5f00\u53d1\u6587\u6863\u300b"),"\u3002"),(0,o.kt)("p",null,(0,o.kt)("img",{src:n(5857).Z})),(0,o.kt)("h3",{id:"5\u56fa\u4ef6\u6253\u5305\u4e0e\u70e7\u5f55"},"5.\u56fa\u4ef6\u6253\u5305\u4e0e\u70e7\u5f55"),(0,o.kt)("p",null,"\u4e00\u5207\u914d\u7f6e\u5c31\u7eea\u540e\uff0c\u53ef\u70b9\u51fb LSTudio \u4e0a\u7684\u3010\u6253\u5305\u3011\u6309\u94ae\uff0c\u5982\u4e0b\u56fe"),(0,o.kt)("p",null,(0,o.kt)("img",{src:n(3367).Z})),(0,o.kt)("p",null,"\u70b9\u51fb\u6253\u5305\uff0c\u5e76\u7b49\u5f85\u6253\u5305\u8fc7\u7a0b\u5b8c\u6bd5\u3002\u4e4b\u540e\u5c06 LSKits \u4e0e\u7535\u8111\u901a\u8fc7 USB \u7ebf\u8fde\u63a5\uff0c\u540c\u65f6\u6309\u4f4f LSKits \u4e0a\u7684\u3010CSK UPDATE\u3011\u952e\uff0c\u6700\u540e\u70b9\u51fb LSTudio \u4e0a\u7684\u3010\u70e7\u5f55\u3011\u952e\uff0c\u8fdb\u5165\u56fa\u4ef6\u70e7\u5f55\u6a21\u5f0f\uff0c\u7b49\u5f85\u56fa\u4ef6\u70e7\u5f55\u5b8c\u6bd5\u540e\uff0c\u91cd\u542f LSKits \u5373\u53ef\u3002"))}d.isMDXComponent=!0},827:function(e,t,n){t.Z=n.p+"assets/images/Create_project-5d11a6ab26b6592e904300ba63f44181.png"},2611:function(e,t,n){t.Z=n.p+"assets/images/I2S_out_chs-f8d6c30c760294ecc24dd75ade072a5b.png"},3043:function(e,t,n){t.Z=n.p+"assets/images/System_mode-786d17771b7f0b028537682e33a32d3d.png"},5857:function(e,t,n){t.Z=n.p+"assets/images/interact-41e3ca0ca6177fb1b743205e73e58969.png"},3367:function(e,t,n){t.Z=n.p+"assets/images/packaging-373b0a615b4bd23b3dd3401606f0ff28.png"}}]);