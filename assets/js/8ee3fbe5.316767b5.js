(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[996],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return s},kt:function(){return d}});var l=t(67294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function r(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);n&&(l=l.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,l)}return t}function p(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?r(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):r(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,l,a=function(e,n){if(null==e)return{};var t,l,a={},r=Object.keys(e);for(l=0;l<r.length;l++)t=r[l],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(l=0;l<r.length;l++)t=r[l],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var i=l.createContext({}),c=function(e){var n=l.useContext(i),t=n;return e&&(t="function"==typeof e?e(n):p(p({},n),e)),t},s=function(e){var n=c(e.components);return l.createElement(i.Provider,{value:n},e.children)},_={inlineCode:"code",wrapper:function(e){var n=e.children;return l.createElement(l.Fragment,{},n)}},u=l.forwardRef((function(e,n){var t=e.components,a=e.mdxType,r=e.originalType,i=e.parentName,s=o(e,["components","mdxType","originalType","parentName"]),u=c(t),d=a,m=u["".concat(i,".").concat(d)]||u[d]||_[d]||r;return t?l.createElement(m,p(p({ref:n},s),{},{components:t})):l.createElement(m,p({ref:n},s))}));function d(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var r=t.length,p=new Array(r);p[0]=u;var o={};for(var i in n)hasOwnProperty.call(n,i)&&(o[i]=n[i]);o.originalType=e,o.mdxType="string"==typeof e?e:a,p[1]=o;for(var c=2;c<r;c++)p[c]=t[c];return l.createElement.apply(null,p)}return l.createElement.apply(null,t)}u.displayName="MDXCreateElement"},80807:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return o},contentTitle:function(){return i},metadata:function(){return c},toc:function(){return s},default:function(){return u}});var l=t(22122),a=t(19756),r=(t(67294),t(3905)),p=["components"],o={},i="LVGL",c={unversionedId:"chips/600X/application/modules/lvgl",id:"chips/600X/application/modules/lvgl",isDocsHomePage:!1,title:"LVGL",description:"\u672c\u8282\u5c06\u4e3b\u8981\u4ecb\u7ecd\u4f7f\u7528\u5728CSK6\u7684Zephyr SDK\u4e2d\uff0c\u5982\u4f55\u4f7f\u7528LVGL\u5bf9\u663e\u793a\u5c4f\u8fdb\u884cGUI\u7684\u8f93\u51fa\uff0c\u901a\u8fc7\u672c\u7ae0\u8282\u6211\u4eec\u5c06\u638c\u63e1\uff1a",source:"@site/docs/chips/600X/application/modules/lvgl.md",sourceDirName:"chips/600X/application/modules",slug:"/chips/600X/application/modules/lvgl",permalink:"/chips/600X/application/modules/lvgl",version:"current",frontMatter:{}},s=[{value:"LVGL\u7b80\u4ecb",id:"lvgl\u7b80\u4ecb",children:[]},{value:"LVGL\u5173\u952e\u6982\u5ff5",id:"lvgl\u5173\u952e\u6982\u5ff5",children:[]},{value:"LVGL Sample\u8fd0\u884c",id:"lvgl-sample\u8fd0\u884c",children:[]},{value:"Sample\u4ee3\u7801\u8bf4\u660e",id:"sample\u4ee3\u7801\u8bf4\u660e",children:[]}],_={toc:s};function u(e){var n=e.components,o=(0,a.Z)(e,p);return(0,r.kt)("wrapper",(0,l.Z)({},_,o,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"lvgl"},"LVGL"),(0,r.kt)("p",null,"\u672c\u8282\u5c06\u4e3b\u8981\u4ecb\u7ecd\u4f7f\u7528\u5728CSK6\u7684Zephyr SDK\u4e2d\uff0c\u5982\u4f55\u4f7f\u7528LVGL\u5bf9\u663e\u793a\u5c4f\u8fdb\u884cGUI\u7684\u8f93\u51fa\uff0c\u901a\u8fc7\u672c\u7ae0\u8282\u6211\u4eec\u5c06\u638c\u63e1\uff1a"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"\u5bf9LVGL\u7684\u57fa\u672c\u8ba4\u8bc6"),(0,r.kt)("li",{parentName:"ul"},"\u5bf9LVGL\u4e00\u4e9b\u5173\u952e\u6982\u5ff5\u7684\u7406\u89e3"),(0,r.kt)("li",{parentName:"ul"},"\u5728CSK6\u5f00\u53d1\u677f\u4e0a\u8fd0\u884c\u4e00\u4e2aLVGL Sample")),(0,r.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,r.kt)("div",{parentName:"div",className:"admonition-heading"},(0,r.kt)("h5",{parentName:"div"},(0,r.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,r.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,r.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,r.kt)("div",{parentName:"div",className:"admonition-content"},(0,r.kt)("p",{parentName:"div"},"\u672c\u7ae0\u8282\u4f7f\u7528\u7684\u5b9e\u9a8c\u786c\u4ef6\u5e73\u53f0\u4e3a ",(0,r.kt)("strong",{parentName:"p"},"\u626b\u63cf\u7b14\u5f00\u53d1\u677f"),"\uff0c\u5bf9\u5e94SDK\u677f\u578b\u4e3a ",(0,r.kt)("strong",{parentName:"p"},"csk6002_9s_nano"),"\uff0c\u5bf9\u5e94\u7684sdk\u7248\u672c\u672a",(0,r.kt)("strong",{parentName:"p"},"csk6 sdk v1.0.2"),"\u4ee5\u4e0a\u7248\u672c\u3002\n\u4f7f\u7528\u7684Sample\u8def\u5f84\u4e3a",(0,r.kt)("inlineCode",{parentName:"p"},"samples/boards/csk6/subsys/display/lvgl"),"\uff0c\u4f7f\u7528\u4ee5\u4e0a\u786c\u4ef6\u5e73\u53f0\u8fdb\u884c\u672c\u5b9e\u9a8c\u65f6\uff0c\u9700\u8981\u51c6\u5907ST7789V\u663e\u793a\u5c4f\u548cSPI\u8f6c\u63a5\u677f\uff0c\u901a\u8fc7\u675c\u90a6\u7ebf\u5b8c\u6210\u63a5\u7ebf\uff0c\u5177\u4f53\u8fde\u63a5\u65b9\u5f0f\u8bf7\u4e0eFAE\u786e\u8ba4\u3002",(0,r.kt)("br",{parentName:"p"}),"\n","\u540e\u7eedNanoKit\u5f00\u53d1\u677f\u5c06\u63d0\u4f9b\u7528\u4e8eLVGL\u5b9e\u9a8c\u7684\u914d\u4ef6\u677f\uff0c\u656c\u8bf7\u671f\u5f85\u3002"))),(0,r.kt)("h2",{id:"lvgl\u7b80\u4ecb"},"LVGL\u7b80\u4ecb"),(0,r.kt)("p",null,"LVGL\u5168\u79f0Light and Versatile Graphics Library\uff0c\u662f\u4e00\u4e2a\u5f00\u6e90\u7684GUI\u5e93\uff0c\u5177\u5907\u8d44\u6e90\u6d88\u8017\u5c0f\u3001\u53ef\u79fb\u690d\u5ea6\u9ad8\u7684\u7279\u70b9\uff0c\u56e0\u6b64\u5f88\u9002\u5408\u5728\u5d4c\u5165\u5f0f\u8bbe\u5907\u4e0a\u4f7f\u7528\u3002"),(0,r.kt)("p",null,"\u4f5c\u4e3a\u4e00\u4e2a\u56fe\u5f62\u5e93\uff0cLVGL\u4e5f\u81ea\u5e26\u4e86\u4e30\u5bcc\u7684GUI\u63a7\u4ef6\u53ef\u4ee5\u4f9b\u5f00\u53d1\u8005\u76f4\u63a5\u4f7f\u7528\uff0c\u5982\u5e38\u7528\u7684\u6309\u94ae\uff0c\u56fe\u8868\uff0c\u5217\u8868\uff0c\u6ed1\u5757\u7b49\uff0c\u6b64\u5916\u8fd8\u652f\u6301\u591a\u79cd\u56fe\u5f62\u52a8\u753b\u7684\u8c03\u7528\uff0c\u76f8\u6bd4\u4f20\u7edf\u9a71\u5c4f\u7ed8\u5236GUI\uff0c\u5f00\u53d1\u8005\u4e0d\u9700\u8981\u5927\u91cf\u7684\u4ee3\u7801\u7f16\u5199\u91cf\u548c\u53cd\u590d\u7684\u6548\u679c\u9a8c\u8bc1\uff0c\u5c31\u53ef\u4ee5\u642d\u5efa\u51fa\u4e00\u4e2a\u6bd4\u8f83\u6ee1\u610f\u7684GUI\u3002"),(0,r.kt)("h2",{id:"lvgl\u5173\u952e\u6982\u5ff5"},"LVGL\u5173\u952e\u6982\u5ff5"),(0,r.kt)("p",null,"\u5728\u5f00\u59cbLVGL\u7684Sample\u5b9e\u9a8c\u524d\uff0c\u8ba9\u6211\u4eec\u5148\u7b80\u5355\u4e86\u89e3\u4e00\u4e0bLVGL\u4e00\u4e9b\u6bd4\u8f83\u91cd\u8981\u7684\u6982\u5ff5\uff0c\u65b9\u4fbf\u5bf9\u540e\u7eedSample\u9a8c\u8bc1\u7684\u7406\u89e3\u3002"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Display\uff1a")," \u5728LVGL\u4e2d\u4e00\u4e2a\u7269\u7406\u663e\u793a\u5668\u5bf9\u5e94\u4e00\u4e2aDisplay\uff0cLVGL\u652f\u6301\u591a\u4e2a\u7269\u7406\u663e\u793a\u5668\uff0c\u5982\u679c\u5f00\u53d1\u7684\u662f\u50cf\u626b\u63cf\u8bcd\u5178\u7b14\u8fd9\u7c7b\u53ea\u6709\u4e00\u4e2a\u5c4f\u5e55\u8bbe\u5907\uff0c\u5219\u5728\u8fd9\u4e2aLVGL\u5de5\u7a0b\u53ea\u6709\u4e00\u4e2aDisplay\u3002"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Screen\uff1a")," \u4e00\u4e2aDisplay\u521b\u5efa\u591a\u4e2aScreen\uff0c\u53ef\u4ee5\u7406\u89e3\u4e3a\u9875\u9762\uff0cScreen\u4e0a\u53ef\u4ee5\u6dfb\u52a0\u5404\u79cdLVGL\u652f\u6301\u7684Widgets(\u7ec4\u4ef6)"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Object\uff1a")," \u662fLVGL\u6784\u6210GUI\u7684\u57fa\u672c\u5355\u4f4d\uff0c\u6bcf\u4e2aWidgets\u90fd\u5c5e\u4e8eObject\uff0c\u4e0a\u9762\u7684Screen(\u9875\u9762)\u4e5f\u662f\u4e00\u4e2aObject\u3002object\u4f7f\u7528\u7236\u5b50\u6811\u5f62\u7ed3\u6784\uff0c\u53ea\u5141\u8bb8\u6709\u4e00\u4e2a\u7236\u8282\u70b9\uff0c\u5141\u8bb8\u6709\u591a\u4e2a\u5b50\u8282\u70b9\u3002screen\u505a\u4e3a\u6839\u8282\u70b9\uff0c\u5141\u8bb8\u6ca1\u6709\u7236\u8282\u70b9\u3002"),(0,r.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,r.kt)("div",{parentName:"div",className:"admonition-heading"},(0,r.kt)("h5",{parentName:"div"},(0,r.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,r.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,r.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,r.kt)("div",{parentName:"div",className:"admonition-content"},(0,r.kt)("p",{parentName:"div"},"\u66f4\u591a\u7684LVGL\u91cd\u8981\u529f\u80fd\u8bf4\u660e\uff0c\u53ef\u8bbf\u95ee ",(0,r.kt)("a",{parentName:"p",href:"https://docs.lvgl.io/7.11/overview/index.html"},"LVGL\u6587\u6863\u4e2d\u5fc3")," \u8fdb\u884c\u4e86\u89e3\u3002"))),(0,r.kt)("h2",{id:"lvgl-sample\u8fd0\u884c"},"LVGL Sample\u8fd0\u884c"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Step 1\uff1a \u521b\u5efaSample")),(0,r.kt)("p",null,"\u901a\u8fc7Lisa\u547d\u4ee4\u521b\u5efa\u9879\u76ee\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"lisa zep create\n")),(0,r.kt)("p",null,"\u4f9d\u6b21\u6309\u4ee5\u4e0b\u76ee\u5f55\u9009\u62e9\u5b8c\u6210adc sample\u7684\u521b\u5efa\uff1a  "),(0,r.kt)("blockquote",null,(0,r.kt)("p",{parentName:"blockquote"},"boards \u2192 csk6 \u2192 subsys \u2192 display \u2192 lvgl")),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Step 2\uff1a \u7ec4\u4ef6\u914d\u7f6e")),(0,r.kt)("p",null,(0,r.kt)("inlineCode",{parentName:"p"},"lvgl")," \u5de5\u7a0b\u901a\u7528\u914d\u7f6e\u6587\u4ef6 ",(0,r.kt)("inlineCode",{parentName:"p"},"prj.conf")," \uff0c\u5728\u6587\u4ef6\u4e2d\u589e\u52a0\u4ee5\u4e0b\u914d\u7f6e\u9009\u9879\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-shell"},"\n    CONFIG_HEAP_MEM_POOL_SIZE=16384\n    CONFIG_MAIN_STACK_SIZE=2048\n\n    CONFIG_DISPLAY=y\n    CONFIG_DISPLAY_LOG_LEVEL_ERR=y\n\n    CONFIG_LOG=y\n    CONFIG_LOG_STRDUP_BUF_COUNT=16\n\n    CONFIG_LVGL=y\n    CONFIG_LVGL_USE_LABEL=y\n    CONFIG_LVGL_USE_CONT=y\n    CONFIG_LVGL_USE_BTN=y\n    CONFIG_LVGL_USE_THEME_MATERIAL=y\n\n    # \u542f\u7528GPIO\u9a71\u52a8(\u5c4f\u5e55\u63a7\u5236\u5f15\u811a\u4f7f\u7528)\n    CONFIG_GPIO=y\n    # \u542f\u7528SPI\u9a71\u52a8(\u5c4f\u5e55\u4f7f\u7528SPI\u4f5c\u4e3a\u6570\u636e\u603b\u7ebf)\n    CONFIG_SPI=y\n    # \u542f\u7528ST7789V\u8bbe\u5907(\u5173\u8054\u8bbe\u5907\u9a71\u52a8\u6587\u4ef6)\n    CONFIG_ST7789V=y\n\n    CONFIG_I2C=y\n    CONFIG_KSCAN_BL6XXX=y\n")),(0,r.kt)("p",null,"\u9488\u5bf9csk6002_9s_nano\u5f00\u53d1\u677f\u7684\u786c\u4ef6\u914d\u7f6e\uff1a\n",(0,r.kt)("inlineCode",{parentName:"p"},"lvgl")," \u5de5\u7a0b\u76ee\u5f55\u4e0b",(0,r.kt)("inlineCode",{parentName:"p"},"boards/csk6002_9s_nano.conf"),"\u589e\u52a0\u5982\u4e0b\u914d\u7f6e\uff1a   "),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-shell"},'    CONFIG_KSCAN=y\n    CONFIG_LVGL_POINTER_KSCAN_SWAP_XY=y\n    CONFIG_LVGL_POINTER_KSCAN=y\n    CONFIG_LVGL_POINTER_KSCAN_DEV_NAME="BL6XXX"\n\n    CONFIG_DISPLAY=y\n    # \u5f00\u53d1\u677f\u4f7f\u7528\u7684\u5c4f\u5e55\u662f ST7789V\n    CONFIG_LVGL_DISPLAY_DEV_NAME="ST7789V"\n    # \u5f00\u53d1\u677f\u4f7f\u7528\u7684\u5c4f\u5e55\u5bbd\n    CONFIG_LVGL_HOR_RES_MAX=320\n    # \u5f00\u53d1\u677f\u4f7f\u7528\u7684\u5c4f\u5e55\u9ad8\n    CONFIG_LVGL_VER_RES_MAX=170\n    CONFIG_LVGL_DPI=100\n')),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Step 3\uff1a \u7f16\u8bd1")),(0,r.kt)("p",null,"\u5728\u5f53\u524d\u5de5\u7a0b\u76ee\u5f55\u4e2d\u6267\u884c ",(0,r.kt)("inlineCode",{parentName:"p"},"lisa zep build -b csk6002_9s_nano")," \u6307\u4ee4\uff0c\u6307\u5b9a\u6211\u4eec\u4f7f\u7528\u7684\u5f00\u53d1\u677f\u677f\u578b\u8fdb\u884c\u7f16\u8bd1\u3002"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Step 4\uff1a \u70e7\u5f55&\u89c2\u5bdf\u5c4f\u5e55\u8f93\u51fa\u5185\u5bb9")),(0,r.kt)("p",null,"\u7f16\u8bd1\u5b8c\u6210\u540e\uff0c\u6267\u884c ",(0,r.kt)("inlineCode",{parentName:"p"},"lisa zep flash --runner pyocd")," \u6307\u4ee4\u8fdb\u884c\u56fa\u4ef6\u70e7\u5f55"),(0,r.kt)("p",null,'\u70e7\u5f55\u5b8c\u6210\u540e\uff0c\u53ef\u89c2\u5bdf\u5230\u8bbe\u5907\u663e\u793a\u5c4f\u51fa\u73b0\u56fe\u50cf\uff0c\u4e2d\u592e\u6709\u4e00\u4e2atext\u5185\u5bb9\u4e3a"Hello world!"\u7684lable\uff0c\u4e0b\u65b9\u4e3a\u4e00\u4e2a\u4e0d\u65ad\u9012\u589e\u7684\u8ba1\u6570\u503c\uff0c\u5982\u56fe\uff1a\n',(0,r.kt)("img",{alt:"image",src:t(49467).Z})),(0,r.kt)("h2",{id:"sample\u4ee3\u7801\u8bf4\u660e"},"Sample\u4ee3\u7801\u8bf4\u660e"),(0,r.kt)("p",null,"LVGL Sample\u7b80\u5355\u5c55\u793a\u4e86object\u7684\u521b\u5efa\u548clable\u63a7\u4ef6\u7684\u4f7f\u7528\uff0c\u4e3b\u8981\u4ee3\u7801\u53ca\u6ce8\u91ca\u5982\u4e0b\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-c"},'    // \u521b\u5efa\u4e24\u4e2alabel\u5bf9\u8c61\n    lv_obj_t *hello_world_label;\n    lv_obj_t *count_label;\n\n    // \u521b\u5efa\u4e00\u4e2adisplay\u8bbe\u5907\n    const struct device *display_dev;\n    \n    // \u8fdb\u884c\u8bbe\u5907\u7684\u7ed1\u5b9a\uff0cCONFIG_LVGL_DISPLAY_DEV_NAME\u5df2\u5728prj.conf\u4e2d\u5b9a\u4e49\u4e3aST7789V\n    display_dev = device_get_binding(CONFIG_LVGL_DISPLAY_DEV_NAME);\n    \n    // \u5c06\u521b\u5efa\u7684\u5bf9\u8c61\u505a\u4e3alable\u4f7f\u7528\uff0c\u8bbe\u7f6etext\u5185\u5bb9\u4e3a"Hello world!"\n    lv_label_set_text(hello_world_label, "Hello world!");\n    // \u5bf9\u8fd9\u4e2alable\u505a\u4f4d\u7f6e\u7684\u6392\u5217\uff0c\u4f4d\u7f6e\u4e3a\u9875\u9762\u6b63\u4e2d\u95f4\uff0c\u540e\u9762\u4e24\u4e2a\u53c2\u6570\u4e3ax/y\u5750\u6807\u504f\u79fb\u91cf\n    lv_obj_align(hello_world_label, NULL, LV_ALIGN_CENTER, 0, 0);\n    \n    // \u5c06count_label\u4f5c\u4e3a\u4e00\u4e2alable\uff0c\u5e76\u5c06\u5176\u7236\u8282\u70b9\u914d\u7f6e\u4e3a\u9ed8\u8ba4\u7684screen\n    count_label = lv_label_create(lv_scr_act(), NULL);\n    // \u5bf9\u8fd9\u4e2alable\u505a\u4f4d\u7f6e\u7684\u6392\u5217\uff0c\u4f4d\u7f6e\u4e3a\u5e95\u90e8\u7684\u4e2d\u95f4\n    lv_obj_align(count_label, NULL, LV_ALIGN_IN_BOTTOM_MID, 0, 0);\n    \n    // \u6267\u884c\u4efb\u52a1handler\uff0c\u8fdb\u884c\u5c4f\u5e55\u5185\u5bb9\u5237\u65b0\n    lv_task_handler();\n    display_blanking_off(display_dev);\n    \n    // \u4e00\u4e2awhile(1)\u5faa\u73af\u4e2d\uff0c\u6bcf\u4e2a1\u79d2\u505a\u4e00\u6b21count_label\u7684text\u7684\u6570\u503c\u9012\u589e1\n    while (1) {\n        if ((count % 100) == 0U) {\n            sprintf(count_str, "%d", count/100U);\n            lv_label_set_text(count_label, count_str);\n        }\n        // \u9700\u8981\u5b9a\u671f\u6267\u884clv_task_handler\n        lv_task_handler();\n        k_sleep(K_MSEC(10));\n        ++count;\n    }\n')),(0,r.kt)("p",null,"\u5728\u672c\u6587\u4e2d\uff0cLVGL Sample\u4ec5\u7b80\u5355\u5c55\u793a\u4e86LVGL\u7684GUI\u529f\u80fd\uff0c\u66f4\u591a\u7684\u63a7\u4ef6\u3001\u9ad8\u7ea7\u529f\u80fd\u5e94\u7528\u7528\u6237\u53ef\u4ee5\u81ea\u884c\u9605\u8bfbLVGL\u5e93\u63a5\u53e3\u8bf4\u660e\u6216\u524d\u5f80 ",(0,r.kt)("a",{parentName:"p",href:"https://docs.lvgl.io/7.11/overview/index.html"},"LVGL\u5b98\u65b9\u6587\u6863")," \u8fdb\u884c\u4e86\u89e3\u3002"))}u.isMDXComponent=!0},49467:function(e,n,t){"use strict";n.Z=t.p+"assets/images/lvgl_helloworld-df31f13f45a4a5ec15e90f070b453019.png"}}]);