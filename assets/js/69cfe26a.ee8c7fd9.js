(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[268],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return c},kt:function(){return d}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),p=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=p(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,l=e.originalType,s=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),m=p(n),d=a,f=m["".concat(s,".").concat(d)]||m[d]||u[d]||l;return n?r.createElement(f,i(i({ref:t},c),{},{components:n})):r.createElement(f,i({ref:t},c))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=n.length,i=new Array(l);i[0]=m;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o.mdxType="string"==typeof e?e:a,i[1]=o;for(var p=2;p<l;p++)i[p]=n[p];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},38657:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return s},metadata:function(){return p},toc:function(){return c},default:function(){return m}});var r=n(22122),a=n(19756),l=(n(67294),n(3905)),i=["components"],o={title:"\u5b89\u88c5",sidebar_position:1},s=void 0,p={unversionedId:"tools/LISA_LPM/installation",id:"tools/LISA_LPM/installation",isDocsHomePage:!1,title:"\u5b89\u88c5",description:"\u83b7\u53d6",source:"@site/docs/tools/LISA_LPM/installation.md",sourceDirName:"tools/LISA_LPM",slug:"/tools/LISA_LPM/installation",permalink:"/tools/LISA_LPM/installation",version:"current",sidebarPosition:1,frontMatter:{title:"\u5b89\u88c5",sidebar_position:1},sidebar:"toolsLISA",next:{title:"\u4ecb\u7ecd",permalink:"/tools/LISA_LPM/awesome"}},c=[{value:"\u83b7\u53d6",id:"\u83b7\u53d6",children:[]},{value:"\u66f4\u65b0",id:"\u66f4\u65b0",children:[{value:"\u65b9\u5f0f\u4e00\uff1a\u7528lisa\u547d\u4ee4",id:"\u65b9\u5f0f\u4e00\uff1a\u7528lisa\u547d\u4ee4",children:[]},{value:"\u65b9\u5f0f\u4e8c\uff1a\u901a\u8fc7npm\u66f4\u65b0",id:"\u65b9\u5f0f\u4e8c\uff1a\u901a\u8fc7npm\u66f4\u65b0",children:[]}]}],u={toc:c};function m(e){var t=e.components,n=(0,a.Z)(e,i);return(0,l.kt)("wrapper",(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h2",{id:"\u83b7\u53d6"},"\u83b7\u53d6"),(0,l.kt)("p",null,"1\u3001\u786e\u4fdd\u4f60\u5b89\u88c5\u4e86\u8f83\u65b0\u7248\u672c\u7684Node.js\u3002\u53ef\u53c2\u8003\u8be5",(0,l.kt)("a",{parentName:"p",href:"./node_installation"},"\u6587\u6863"),"\u8fdb\u884c\u5b89\u88c5\u3002"),(0,l.kt)("p",null,"2\u3001\u5168\u5c40\u5b89\u88c5yarn\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"npm install yarn -g\n")),(0,l.kt)("p",null,"3\u3001\u5168\u5c40\u5b89\u88c5lisa\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"npm install @listenai/lisa -g\n")),(0,l.kt)("p",null,"\u73b0\u5728\uff0c\u8fd0\u884c\u4e0blisa\u7684\u547d\u4ee4\u770b\u770b\u5427~"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"lisa -v\n")),(0,l.kt)("p",null,"\u5c1d\u8bd5\u767b\u5f55\u4f60\u7684\u8046\u601d\u8d26\u53f7~"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"lisa login\n")),(0,l.kt)("h2",{id:"\u66f4\u65b0"},"\u66f4\u65b0"),(0,l.kt)("p",null,"\u82e5\u4f60\u5df2\u7ecf\u5b89\u88c5\u4e86 lisa \uff0c\u540c\u6837\u6709\u4e24\u79cd\u65b9\u5f0f\u8fdb\u884c\u66f4\u65b0\u3002"),(0,l.kt)("h3",{id:"\u65b9\u5f0f\u4e00\uff1a\u7528lisa\u547d\u4ee4"},"\u65b9\u5f0f\u4e00\uff1a\u7528lisa\u547d\u4ee4"),(0,l.kt)("p",null,"lisa \u63d0\u4f9b\u4e86\u66f4\u65b0\u547d\u4ee4\uff0c\u76f4\u63a5\u6267\u884c\u80fd\u5c06\u5f53\u524d\u73af\u5883\u7684lisa\u5347\u7ea7\u5230latest\u7248\u672c\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"lisa update\n")),(0,l.kt)("h3",{id:"\u65b9\u5f0f\u4e8c\uff1a\u901a\u8fc7npm\u66f4\u65b0"},"\u65b9\u5f0f\u4e8c\uff1a\u901a\u8fc7npm\u66f4\u65b0"),(0,l.kt)("p",null,"\u6267\u884cnpm\u5168\u5c40\u5b89\u88c5\u547d\u4ee4\uff0c\u540c\u6837\u80fd\u5c06lisa\u5347\u7ea7\u5230latest\u7248\u672c\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"npm install @listenai/lisa -g\n")),(0,l.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,l.kt)("div",{parentName:"div",className:"admonition-heading"},(0,l.kt)("h5",{parentName:"div"},(0,l.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,l.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,l.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"info")),(0,l.kt)("div",{parentName:"div",className:"admonition-content"},(0,l.kt)("p",{parentName:"div"},"\u82e5\u7531\u4e8e\u989d\u5916\u539f\u56e0\u5bfc\u81f4install\u5931\u8d25\uff0c\u53ef\u5148\u5378\u8f7d\u540e\uff0c\u91cd\u65b0install"))),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"npm uninstall @listenai/lisa -g\n")))}m.isMDXComponent=!0}}]);