(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[6700],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return c},kt:function(){return d}});var l=t(67294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);n&&(l=l.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,l)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function a(e,n){if(null==e)return{};var t,l,r=function(e,n){if(null==e)return{};var t,l,r={},i=Object.keys(e);for(l=0;l<i.length;l++)t=i[l],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(l=0;l<i.length;l++)t=i[l],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var p=l.createContext({}),s=function(e){var n=l.useContext(p),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},c=function(e){var n=s(e.components);return l.createElement(p.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return l.createElement(l.Fragment,{},n)}},m=l.forwardRef((function(e,n){var t=e.components,r=e.mdxType,i=e.originalType,p=e.parentName,c=a(e,["components","mdxType","originalType","parentName"]),m=s(t),d=r,_=m["".concat(p,".").concat(d)]||m[d]||u[d]||i;return t?l.createElement(_,o(o({ref:n},c),{},{components:t})):l.createElement(_,o({ref:n},c))}));function d(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var i=t.length,o=new Array(i);o[0]=m;var a={};for(var p in n)hasOwnProperty.call(n,p)&&(a[p]=n[p]);a.originalType=e,a.mdxType="string"==typeof e?e:r,o[1]=a;for(var s=2;s<i;s++)o[s]=t[s];return l.createElement.apply(null,o)}return l.createElement.apply(null,t)}m.displayName="MDXCreateElement"},12639:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return a},contentTitle:function(){return p},metadata:function(){return s},toc:function(){return c},default:function(){return m}});var l=t(22122),r=t(19756),i=(t(67294),t(3905)),o=["components"],a={},p="\u8f6e\u8be2",s={unversionedId:"chips/600X/application/kernel/sync-poll",id:"chips/600X/application/kernel/sync-poll",isDocsHomePage:!1,title:"\u8f6e\u8be2",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/kernel/sync-poll.md",sourceDirName:"chips/600X/application/kernel",slug:"/chips/600X/application/kernel/sync-poll",permalink:"/chips/600X/application/kernel/sync-poll",version:"current",frontMatter:{}},c=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u5e38\u7528API\u63a5\u53e3",id:"\u5e38\u7528api\u63a5\u53e3",children:[]},{value:"Poll\u7684\u4f7f\u7528",id:"poll\u7684\u4f7f\u7528",children:[]}],u={toc:c};function m(e){var n=e.components,t=(0,r.Z)(e,o);return(0,i.kt)("wrapper",(0,l.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"\u8f6e\u8be2"},"\u8f6e\u8be2"),(0,i.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,i.kt)("p",null,"\u672c\u7ae0\u8282\u5728\u4e0a\u4e2a\u7ae0\u8282-\u540c\u6b65\u4e4b\u4e92\u65a5\u91cf\u57fa\u7840\u4e0a\u8bb2\u89e3\u8f6e\u8be2\u7684\u4f7f\u7528\u573a\u666f\u548c\u4f7f\u7528\u65b9\u6cd5\uff0c\u901a\u8fc7\u672c\u7ae0\u8282\u5b66\u4e60\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u4e86\u89e3\u5230\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u8f6e\u8be2\u7684\u57fa\u672c\u4fe1\u606f\u548c\u4f7f\u7528\u573a\u666f"),(0,i.kt)("li",{parentName:"ul"},"\u8f6e\u8be2\u7684\u4f7f\u7528\u65b9\u6cd5")),(0,i.kt)("p",null,"\u8f6e\u8be2(poll)\u662f\u4e00\u4e2a\u6bd4\u8f83\u7279\u6b8a\u7684\u5185\u6838\u5bf9\u8c61\uff0cpolling API\u5141\u8bb8\u4e00\u4e2a\u7ebf\u7a0b\u7b49\u5f85\u4e00\u4e2a\u6216\u8005\u591a\u4e2a\u6761\u4ef6\u6ee1\u8db3\u3002\u652f\u6301\u7684\u6761\u4ef6\u7c7b\u578b\u53ea\u80fd\u662f\u5185\u6838\u5bf9\u8c61\uff0c\u53ef\u4ee5\u662fSemaphore, FIFO, poll signal\u4e09\u79cd\u3002\n\u4f8b\u5982\u4e00\u4e2a\u7ebf\u7a0b\u4f7f\u7528polling API\u540c\u65f6\u7b49\u5f85\u591a\u4e2asemaphore\uff0c\u53ea\u8981\u5176\u4e2d\u4e00\u4e2asemaphore\u89e6\u53d1\u65f6polling API\u5c31\u4f1a\u5f97\u5230\u901a\u77e5\u3002\npoll\u5177\u6709\u4ee5\u4e0b\u7279\u6027\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u5f53\u4e00\u4e2a\u7ebf\u7a0b\u7b49\u5f85\u591a\u4e2a\u89e6\u53d1\u6761\u4ef6\u65f6\uff0c\u53ea\u8981\u6709\u4e00\u4e2a\u6761\u4ef6\u6ee1\u8db3k_poll\u5c31\u4f1a\u8fd4\u56de\u3002"),(0,i.kt)("li",{parentName:"ul"},"\u5f53semaphore\u6216FIFO\u6ee1\u8db3\u6761\u4ef6\u540e, k_poll\u53ea\u662f\u63a5\u5230\u901a\u77e5\u8fd4\u56de\uff0c\u7ebf\u7a0b\u5e76\u672a\u83b7\u53d6\u5230semaphore\u6216FIFO, \u8fd8\u9700\u8981\u4f7f\u7528\u4ee3\u7801\u4e3b\u52a8\u83b7\u53d6\u3002")),(0,i.kt)("h2",{id:"\u5e38\u7528api\u63a5\u53e3"},"\u5e38\u7528API\u63a5\u53e3"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"/*\u521d\u59cb\u5316\u4e00\u4e2ak_poll_event\u5b9e\u4f8b*/\nvoid k_poll_event_init(struct k_poll_event * event,\nuint32_t type, int mode, void * obj )   \n/*\u8f6e\u8be2k_poll_event\u5b9e\u4f8b,\u7b49\u5f85\u89e6\u53d1\u6761\u4ef6*/\nint k_poll(struct k_poll_event * events, int num_events, k_timeout_t timeout)   \n/*\u521d\u59cb\u5316poll signal\u4fe1\u53f7\uff0c\u4f5c\u4e3apoll event\u7684\u89e6\u53d1*/\nvoid k_poll_signal_init(struct k_poll_signal * sig) \n/*\u53d1\u9001poll signal\u4fe1\u53f7*/\nint k_poll_signal_raise(struct k_poll_signal * sig,\nint result) \n/*\u6e05\u9664signal\u4fe1\u53f7\uff0c\u5982\u679csignal\u88ab\u53d1\u9001\uff0c\u4f46\u8fd8\u672a\u88abpoll\u524d\uff0c\u90fd\u53ef\u4ee5\u4f7f\u7528\u8be5API reset\u6e05\u9664*/\nvoid k_poll_signal_reset(struct k_poll_signal * sig)\n\n/*\u83b7\u53d6signal\u4fe1\u53f7\u7684\u72b6\u6001\u548c\u503c*/\nvoid k_poll_signal_check(struct k_poll_signal * sig, unsigned int * signaled, int * result) \n\n")),(0,i.kt)("p",null,"\u66f4\u591a",(0,i.kt)("inlineCode",{parentName:"p"},"Poll API"),"\u63a5\u53e3 \u53ef\u4ee5\u5728zephyr\u5b98\u7f51",(0,i.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__poll__apis.html"},"Async polling APIs"),"\u4e2d\u627e\u5230\u3002"),(0,i.kt)("h2",{id:"poll\u7684\u4f7f\u7528"},"Poll\u7684\u4f7f\u7528"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u7ebf\u7a0bA poll\u4fe1\u53f7\u662f\u5426\u53d1\u9001"),"    "),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},'/*\u5b9a\u4e49\u4e00\u4e2apoll signal\u4fe1\u53f7*/\nstruct k_poll_signal signal;\n\nvoid thread_A(void){\n    /*\u521d\u59cb\u5316\u4fe1\u53f7\uff0c\u5e76\u5c06\u5176\u4f5c\u4e3apoll\u6761\u4ef6*/\n    k_poll_signal_init(&signal);\n    \n    /*\u521d\u59cb\u5316poll event\u6761\u4ef6*/\n    struct k_poll_event events[1] = {\n        K_POLL_EVENT_INITIALIZER(K_POLL_TYPE_SIGNAL,\n                                 K_POLL_MODE_NOTIFY_ONLY,\n                                 &signal),\n    };\n\n    while(1){\n            k_poll(events, 1, K_FOREVER);\n\n            if (events[0].signal->result == 0x1337) {\n                printk("get signal");\n            } else {\n                 printk("weird error");\n            }\n\n            /*\u5728\u5faa\u73af\u4e2d\u8c03\u7528 k_poll() \u65f6\uff0c\u7528\u6237\u5fc5\u987b\u5c06event\u7684state\u6001\u91cd\u7f6e\u4e3aK_POLL_STATE_NOT_READY*/\n            events[0].signal->signaled = 0;\n            events[0].state = K_POLL_STATE_NOT_READY;\n    }\n}\n')),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u7ebf\u7a0b B \u53d1\u9001signal\u4fe1\u53f7\u89e6\u53d1event")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"void thread_B(void)\n{\n    k_poll_signal_raise(&signal, 0x1337);\n}\n")),(0,i.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,i.kt)("div",{parentName:"div",className:"admonition-heading"},(0,i.kt)("h5",{parentName:"div"},(0,i.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,i.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,i.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,i.kt)("div",{parentName:"div",className:"admonition-content"},(0,i.kt)("p",{parentName:"div"},"\u672c\u7ae0\u8282\u540c\u6b65\u4e4b\u8f6e\u8be2\u5185\u5bb9\u4e0d\u5728",(0,i.kt)("inlineCode",{parentName:"p"},"csk6 sdk"),"\u4e2d\u63d0\u4f9b\u5177\u4f53\u7684\u5b9e\u73b0sample\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u6839\u636e\u8f6e\u8be2\u7684\u4f7f\u7528\u53c2\u8003\u5c1d\u8bd5\u5b8c\u6210\u5e94\u7528\u5f00\u53d1\u548c\u63a2\u7d22\uff0c\u4ee5\u63d0\u5347\u5f00\u53d1\u719f\u7ec3\u5ea6\u3002"))))}m.isMDXComponent=!0}}]);