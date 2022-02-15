(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[8986],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return c},kt:function(){return d}});var r=t(67294);function l(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function o(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function a(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?o(Object(t),!0).forEach((function(n){l(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,r,l=function(e,n){if(null==e)return{};var t,r,l={},o=Object.keys(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||(l[t]=e[t]);return l}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(l[t]=e[t])}return l}var i=r.createContext({}),p=function(e){var n=r.useContext(i),t=n;return e&&(t="function"==typeof e?e(n):a(a({},n),e)),t},c=function(e){var n=p(e.components);return r.createElement(i.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,l=e.mdxType,o=e.originalType,i=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),m=p(t),d=l,f=m["".concat(i,".").concat(d)]||m[d]||u[d]||o;return t?r.createElement(f,a(a({ref:n},c),{},{components:t})):r.createElement(f,a({ref:n},c))}));function d(e,n){var t=arguments,l=n&&n.mdxType;if("string"==typeof e||l){var o=t.length,a=new Array(o);a[0]=m;var s={};for(var i in n)hasOwnProperty.call(n,i)&&(s[i]=n[i]);s.originalType=e,s.mdxType="string"==typeof e?e:l,a[1]=s;for(var p=2;p<o;p++)a[p]=t[p];return r.createElement.apply(null,a)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},31045:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return s},contentTitle:function(){return i},metadata:function(){return p},toc:function(){return c},default:function(){return m}});var r=t(22122),l=t(19756),o=(t(67294),t(3905)),a=["components"],s={title:"node\u73af\u5883\u642d\u5efa"},i=void 0,p={unversionedId:"tools/LISA_LPM/node_installation",id:"tools/LISA_LPM/node_installation",isDocsHomePage:!1,title:"node\u73af\u5883\u642d\u5efa",description:"Windows",source:"@site/docs/tools/LISA_LPM/node_installation.md",sourceDirName:"tools/LISA_LPM",slug:"/tools/LISA_LPM/node_installation",permalink:"/tools/LISA_LPM/node_installation",version:"current",frontMatter:{title:"node\u73af\u5883\u642d\u5efa"},sidebar:"toolsLISA",previous:{title:"\u5f00\u53d1\u8005\u65b0\u624b\u6559\u7a0b\uff1a\u4ece\u96f6\u5f00\u53d1\u4e00\u4e2aLPM\u5305",permalink:"/tools/LISA_LPM/other/tutorial"},next:{title:"\u5347\u7ea7\uff1a\u4ece1.0\u5347\u7ea7\u52302.0",permalink:"/tools/LISA_LPM/update2"}},c=[{value:"Windows",id:"windows",children:[]},{value:"macOS \u6216 linux",id:"macos-\u6216-linux",children:[]}],u={toc:c};function m(e){var n=e.components,t=(0,l.Z)(e,a);return(0,o.kt)("wrapper",(0,r.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h3",{id:"windows"},"Windows"),(0,o.kt)("p",null,"\u6253\u5f00",(0,o.kt)("a",{parentName:"p",href:"https://nodejs.org/zh-cn/"},"node\u5b98\u7f51"),"\uff0c\u9009\u62e9",(0,o.kt)("strong",{parentName:"p"},"\u957f\u671f\u7ef4\u62a4\u7248(LTS)16.13.0"),"\u8fdb\u884c\u4e0b\u8f7d\u5b89\u88c5\u3002"),(0,o.kt)("p",null,"\u6216\u76f4\u63a5\u70b9\u51fb\u4e0b\u8f7d\u5730\u5740\uff1a"),(0,o.kt)("p",null,(0,o.kt)("a",{parentName:"p",href:"https://cdn.iflyos.cn/public/node/v16.13.0/node-v16.13.0-x64.msi"},"node-v16.13.0_Windows(x64)64\u4f4d")),(0,o.kt)("p",null,(0,o.kt)("a",{parentName:"p",href:"https://cdn.iflyos.cn/public/node/v16.13.0/node-v16.13.0-x86.msi"},"node-v16.13.0_Windows(x86)32\u4f4d")),(0,o.kt)("p",null,"\u4e0b\u8f7d\u540e\u70b9\u51fb\u5b89\u88c5\u5373\u53ef\u3002"),(0,o.kt)("p",null,"\u987a\u5229\u5b89\u88c5\u540e\uff0c\u9a8c\u8bc1\u672c\u5730\u7684node\u73af\u5883\u5427~"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"node -v\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"npm -v\n")),(0,o.kt)("h3",{id:"macos-\u6216-linux"},"macOS \u6216 linux"),(0,o.kt)("p",null,"\u7531\u4e8emacOS pkg\u4e00\u952e\u5b89\u88c5\u5b58\u5728\u6743\u9650\u76f8\u5173\u7684\u95ee\u9898\uff0c\u4ee5\u53calinux\u7684\u642d\u5efanode\u73af\u5883\u8f83\u4e3a\u590d\u6742\uff0c\u6b64\u5904\u5efa\u8bae\u4f7f\u7528nvm\u6765\u5b89\u88c5node\u3002"),(0,o.kt)("h4",{id:"\u5b89\u88c5nvm"},"\u5b89\u88c5nvm"),(0,o.kt)("p",null,"\u4f7f\u7528 curl"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash\n")),(0,o.kt)("p",null,"\u6216 wget"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash\n")),(0,o.kt)("p",null,"\u5b89\u88c5\u540e\uff0c\u6267\u884c"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},'export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"\n[ -s "$NVM_DIR/nvm.sh" ] && \\. "$NVM_DIR/nvm.sh" # This loads nvm\n')),(0,o.kt)("p",null,"\u4e0a\u8ff0\u90fd\u987a\u5229\u5b8c\u6210\u540e\uff0c\u91cd\u65b0\u6253\u5f00\u4e2a\u65b0\u7684\u7ec8\u7aef\uff0c\u5c1d\u8bd5\u9a8c\u8bc1\u5b89\u88c5\u7ed3\u679c\u3002"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"nvm\n")),(0,o.kt)("h4",{id:"\u4f7f\u7528nvm\u5b89\u88c5node"},"\u4f7f\u7528nvm\u5b89\u88c5node"),(0,o.kt)("p",null,"\u6267\u884c"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"nvm install 16.13.0\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"nvm use 16.13.0\n")),(0,o.kt)("p",null,"\u987a\u5229\u6267\u884c\u540e\uff0c\u9a8c\u8bc1\u672c\u5730\u7684node\u73af\u5883\u5427~"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"node -v\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell"},"npm -v\n")))}m.isMDXComponent=!0}}]);