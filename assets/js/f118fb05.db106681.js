(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[9800],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return c},kt:function(){return k}});var a=t(67294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function l(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?l(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):l(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},l=Object.keys(e);for(a=0;a<l.length;a++)t=l[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(a=0;a<l.length;a++)t=l[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var d=a.createContext({}),p=function(e){var n=a.useContext(d),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},c=function(e){var n=p(e.components);return a.createElement(d.Provider,{value:n},e.children)},s={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},u=a.forwardRef((function(e,n){var t=e.components,r=e.mdxType,l=e.originalType,d=e.parentName,c=i(e,["components","mdxType","originalType","parentName"]),u=p(t),k=r,m=u["".concat(d,".").concat(k)]||u[k]||s[k]||l;return t?a.createElement(m,o(o({ref:n},c),{},{components:t})):a.createElement(m,o({ref:n},c))}));function k(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var l=t.length,o=new Array(l);o[0]=u;var i={};for(var d in n)hasOwnProperty.call(n,d)&&(i[d]=n[d]);i.originalType=e,i.mdxType="string"==typeof e?e:r,o[1]=i;for(var p=2;p<l;p++)o[p]=t[p];return a.createElement.apply(null,o)}return a.createElement.apply(null,t)}u.displayName="MDXCreateElement"},52194:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return i},contentTitle:function(){return d},metadata:function(){return p},toc:function(){return c},default:function(){return u}});var a=t(22122),r=t(19756),l=(t(67294),t(3905)),o=["components"],i={},d="\u591a\u7ebf\u7a0b",p={unversionedId:"chips/600X/application/kernel/multithread",id:"chips/600X/application/kernel/multithread",isDocsHomePage:!1,title:"\u591a\u7ebf\u7a0b",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/kernel/multithread.md",sourceDirName:"chips/600X/application/kernel",slug:"/chips/600X/application/kernel/multithread",permalink:"/chips/600X/application/kernel/multithread",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"Timer\u5b9a\u65f6\u5668",permalink:"/chips/600X/application/kernel/timer"},next:{title:"\u7ebf\u7a0b\u95f4\u540c\u6b65\u4e4b\u4fe1\u53f7\u91cf",permalink:"/chips/600X/application/kernel/sync_semaphore"}},c=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u591a\u7ebf\u7a0b\u7b80\u4ecb",id:"\u591a\u7ebf\u7a0b\u7b80\u4ecb",children:[]},{value:"\u521b\u5efa\u9879\u76ee",id:"\u521b\u5efa\u9879\u76ee",children:[{value:"\u9879\u76ee\u914d\u7f6e",id:"\u9879\u76ee\u914d\u7f6e",children:[]},{value:"\u5e38\u7528API\u63a5\u53e3",id:"\u5e38\u7528api\u63a5\u53e3",children:[]},{value:"\u4ee3\u7801\u5b9e\u73b0",id:"\u4ee3\u7801\u5b9e\u73b0",children:[]}]},{value:"\u7f16\u8bd1\u548c\u70e7\u5f55",id:"\u7f16\u8bd1\u548c\u70e7\u5f55",children:[]}],s={toc:c};function u(e){var n=e.components,i=(0,r.Z)(e,o);return(0,l.kt)("wrapper",(0,a.Z)({},s,i,{components:n,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"\u591a\u7ebf\u7a0b"},"\u591a\u7ebf\u7a0b"),(0,l.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,l.kt)("p",null,"\u5728\u5e94\u7528\u5f00\u53d1\u7684\u8fc7\u7a0b\u4e2d\u7ecf\u5e38\u4f1a\u4f7f\u7528\u5230\u591a\u7ebf\u7a0b\uff0c\u914d\u5408\u5185\u6838\u63d0\u4f9b\u7684\u540c\u6b65\u673a\u5236\uff0c\u53ef\u4ee5\u5b9e\u73b0\u591a\u7ebf\u7a0b\u4e4b\u95f4\u7684\u540c\u6b65\uff0c\u4ece\u800c\u5b9e\u73b0\u66f4\u52a0\u7075\u6d3b\u7684\u5e94\u7528\u4e1a\u52a1\u903b\u8f91\u3002"),(0,l.kt)("p",null,"\u672c\u7ae0\u8282\u7ed9\u5f00\u53d1\u8005\u4ecb\u7ecdZephyr\u7cfb\u7edf\u7ebf\u7a0b\u540c\u6b65\u4e2d\u591a\u7ebf\u7a0b\u7684\u5b9e\u73b0\uff0c\u901a\u8fc7\u672c\u7ae0\u8282\u5b66\u4e60\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u4e86\u89e3\u5230\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u591a\u7ebf\u7a0b\u57fa\u672c\u4fe1\u606f\u548c\u4f7f\u7528\u573a\u666f"),(0,l.kt)("li",{parentName:"ul"},"\u591a\u7ebf\u7a0b\u7684\u5b9e\u73b0\u65b9\u6cd5")),(0,l.kt)("h2",{id:"\u591a\u7ebf\u7a0b\u7b80\u4ecb"},"\u591a\u7ebf\u7a0b\u7b80\u4ecb"),(0,l.kt)("p",null,"\u591a\u7ebf\u7a0b\u5728\u9879\u76ee\u4e2d\u4e3b\u8981\u7528\u6765\u89e3\u51b3\u5e76\u53d1\u4efb\u52a1\u6267\u884c\uff0c\u672c\u7ae0\u8282\u5c06\u901a\u8fc7\u4e00\u4e2a\u7b80\u5355\u7684\u793a\u4f8b\u6765\u8bf4\u660e\u591a\u7ebf\u7a0b\u7684\u5b9e\u73b0\uff0c\u793a\u4f8b\u4e2d\u521b\u5efa\u4e24\u4e2a\u7ebf\u7a0b",(0,l.kt)("inlineCode",{parentName:"p"},"threadA"),"\u548c",(0,l.kt)("inlineCode",{parentName:"p"},"threadB"),"\uff0c\u5e76\u5728\u6bcf\u4e2a\u7ebf\u7a0b\u4e2d\u5206\u522b\u5faa\u73af\u6253\u5370",(0,l.kt)("inlineCode",{parentName:"p"},"Hello World"),"\u3002"),(0,l.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,l.kt)("div",{parentName:"div",className:"admonition-heading"},(0,l.kt)("h5",{parentName:"div"},(0,l.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,l.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,l.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,l.kt)("div",{parentName:"div",className:"admonition-content"},(0,l.kt)("p",{parentName:"div"},"\u672c\u7ae0\u8282\u9700\u8981\u5f00\u53d1\u8005\u57fa\u4e8ecsk6 sdk\u63d0\u4f9b\u7684",(0,l.kt)("inlineCode",{parentName:"p"},"hello_world"),"\u793a\u4f8b\u57fa\u7840\u4e0a\u5b9e\u73b0\u591a\u7ebf\u7a0b\u5f00\u53d1\uff0c\u4ee5\u589e\u5f3a\u5bf9csk6 sdk\u7684\u4e86\u89e3\u548c\u63d0\u5347\u5b9e\u64cd\u80fd\u529b\u3002"))),(0,l.kt)("h2",{id:"\u521b\u5efa\u9879\u76ee"},"\u521b\u5efa\u9879\u76ee"),(0,l.kt)("p",null,"\u901a\u8fc7Lisa\u547d\u4ee4\u521b\u5efa",(0,l.kt)("inlineCode",{parentName:"p"},"hello_world"),"\u9879\u76ee\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep create\n")),(0,l.kt)("p",null,(0,l.kt)("img",{src:t(93279).Z})),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"sample \u2192 hello_world\n\u5b8c\u6210",(0,l.kt)("inlineCode",{parentName:"p"},"hello_world"),"\u9879\u76ee\u521b\u5efa\u540e\uff0c\u53ef\u53c2\u8003\u4ee5\u4e0b\u7ae0\u8282",(0,l.kt)("strong",{parentName:"p"},"API\u63a5\u53e3"),"\u548c",(0,l.kt)("strong",{parentName:"p"},"\u4ee3\u7801\u5b9e\u73b0"),"\u5b8c\u6210\u591a\u7ebf\u7a0b\u793a\u4f8b\u7684\u5b9e\u73b0\u3002")),(0,l.kt)("h3",{id:"\u9879\u76ee\u914d\u7f6e"},"\u9879\u76ee\u914d\u7f6e"),(0,l.kt)("p",null,"\u5728 ",(0,l.kt)("inlineCode",{parentName:"p"},"prj.conf")," \u6587\u4ef6\u4e2d\u589e\u52a0\u4ee5\u4e0b\u914d\u7f6e\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"CONFIG_STDOUT_CONSOLE=y\n# enable to use thread names\nCONFIG_THREAD_NAME=y\nCONFIG_SCHED_CPU_MASK=y\n")),(0,l.kt)("h3",{id:"\u5e38\u7528api\u63a5\u53e3"},"\u5e38\u7528API\u63a5\u53e3"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-c"},"/*\u521b\u5efa\u7ebf\u7a0b*/\nk_tid_t k_thread_create(struct k_thread * new_thread, k_thread_stack_t * stack, size_t stack_size, \nk_thread_entry_t entry, void * p1, void * p2, void * p3, int prio, uint32_t options, k_timeout_t delay)\n\u53c2\u6570\u8bf4\u660e\uff1a\nnew_thread  \u7ebf\u7a0b\u5bf9\u8c61\nstack       \u6307\u5411\u5806\u6808\u7a7a\u95f4\u7684\u6307\u9488\nstack_size  \u4ee5\u5b57\u8282\u4e3a\u5355\u4f4d\u7684\u5806\u6808\u5927\u5c0f\nentry       \u7ebf\u7a0b\u5165\u53e3\u51fd\u6570\np1          \u7b2c\u4e00\u4e2a\u5165\u53e3\u70b9\u53c2\u6570\np2          \u7b2c\u4e8c\u4e2a\u5165\u53e3\u70b9\u53c2\u6570  \np3          \u7b2c\u4e09\u4e2a\u5165\u53e3\u70b9\u53c2\u6570\nprio        \u7ebf\u7a0b\u4f18\u5148\u7ea7\noptions     \u7ebf\u7a0b\u9009\u9879\ndelay       \u8c03\u5ea6\u5ef6\u8fdf \n\n/*\u542f\u52a8\u7ebf\u7a0b*/\nvoid k_thread_start(k_tid_t thread)\n/*\u8bbe\u7f6e\u7ebf\u7a0b\u540d\u79f0*/\nint k_thread_name_set(k_tid_t thread, const char * str)\n")),(0,l.kt)("p",null,"\u66f4\u591a ",(0,l.kt)("strong",{parentName:"p"},"\u7ebf\u7a0bthread API")," \u4f7f\u7528\u65b9\u6cd5\u53ef\u4ee5\u53c2\u7167Zephyr\u5b98\u7f51",(0,l.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__thread__apis.html"},"Thread APIS"),"\u3002"),(0,l.kt)("h3",{id:"\u4ee3\u7801\u5b9e\u73b0"},"\u4ee3\u7801\u5b9e\u73b0"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-c"},'/* \u7ebf\u7a0b\u5806\u6808\u7a7a\u95f4 */\n#define STACKSIZE 1024\n/* \u7ebf\u7a0b\u4f18\u5148\u7ea7 */\n#define PRIORITY 7\n\n\n/* \u7ebf\u7a0bA */\nvoid threadA(void *dummy1, void *dummy2, void *dummy3)\n{\n    /* \u4f5c\u4e3a\u793a\u4f8b\uff0c\u672c\u7ebf\u7a0b\u6ca1\u7528\u5230dummy\u8fd9\u4e09\u4e2a\u53c2\u6570\uff0c\u4f7f\u7528ARG_UNUSED\u8fdb\u884c\u58f0\u660e\uff0c\u53ef\u907f\u514d\u7f16\u8bd1\u65f6\u63d0\u793aWarn */\n    ARG_UNUSED(dummy1);\n    ARG_UNUSED(dummy2);\n    ARG_UNUSED(dummy3);\n\n    while (1){\n        /* \u6bcf\u9694\u4e24\u79d2\u6253\u5370\u4e00\u6b21Hello World */\n        k_msleep(1000);\n        printk("threadA: Hello World on %s!\\n", CONFIG_BOARD);\n    }\n    \n}\n\n/* \u7ebf\u7a0bB */\nvoid threadB(void *dummy1, void *dummy2, void *dummy3)\n{\n\n    ARG_UNUSED(dummy1);\n    ARG_UNUSED(dummy2);\n    ARG_UNUSED(dummy3);\n\n     while (1){\n        k_msleep(2000);\n        printk("threadB: Hello World on %s!\\n", CONFIG_BOARD);\n    }\n}\n\n/* \u5b9a\u4e49\u7ebf\u7a0b\u5806\u6808\u7a7a\u95f4 */\nK_THREAD_STACK_DEFINE(threadA_stack_area, STACKSIZE);\nstatic struct k_thread threadA_data;\n\nK_THREAD_STACK_DEFINE(threadB_stack_area, STACKSIZE);\nstatic struct k_thread threadB_data;\n\n\nvoid main(void)\n{\n    /* \u521b\u5efa\u7ebf\u7a0b */\n    k_thread_create(&threadA_data, threadA_stack_area,\n            K_THREAD_STACK_SIZEOF(threadA_stack_area),\n            threadA, NULL, NULL, NULL,\n            PRIORITY, 0, K_FOREVER);\n    k_thread_name_set(&threadA_data, "thread_a");\n\n    k_thread_create(&threadB_data, threadB_stack_area,\n            K_THREAD_STACK_SIZEOF(threadB_stack_area),\n            threadB, NULL, NULL, NULL,\n            PRIORITY, 0, K_FOREVER);\n    k_thread_name_set(&threadB_data, "thread_b");\n\n    /* \u542f\u52a8\u7ebf\u7a0b */\n    k_thread_start(&threadA_data);\n    k_thread_start(&threadB_data);\n}\n')),(0,l.kt)("h2",{id:"\u7f16\u8bd1\u548c\u70e7\u5f55"},"\u7f16\u8bd1\u548c\u70e7\u5f55"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u7f16\u8bd1")," ")),(0,l.kt)("p",null,"\u5728app\u6839\u76ee\u5f55\u4e0b\u901a\u8fc7\u4ee5\u4e0b\u6307\u4ee4\u5b8c\u6210\u7f16\u8bd1\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep build -b csk6002_9s_nano\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u70e7\u5f55"),"   ")),(0,l.kt)("p",null,(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u901a\u8fc7USB\u8fde\u63a5PC\uff0c\u901a\u8fc7\u70e7\u5f55\u6307\u4ee4\u5f00\u59cb\u70e7\u5f55\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep flash --runner pyocd\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u67e5\u770b\u7ed3\u679c"),"  ")),(0,l.kt)("p",null,"\u53ef\u901a\u8fc7lisa\u63d0\u4f9b\u7684",(0,l.kt)("inlineCode",{parentName:"p"},"lisa term"),"\u547d\u4ee4\u67e5\u770b\u65e5\u5fd7\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa term\n")),(0,l.kt)("p",null,"\u6216\u8005\u5c06",(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u7684\u65e5\u5fd7\u4e32\u53e3",(0,l.kt)("inlineCode",{parentName:"p"},"A03 TX A02 RX"),"\u63a5\u4e32\u53e3\u677f\u8fde\u63a5\u7535\u8111\uff0c\u5728\u7535\u8111\u7aef\u4f7f\u7528\u4e32\u53e3\u8c03\u8bd5\u52a9\u624b\u67e5\u770b\u65e5\u5fd7\uff0c\u6ce2\u7279\u7387\u4e3a115200\u3002"),(0,l.kt)("p",null,"\u9884\u671f\u65e5\u5fd7\u8f93\u51fa\u7ed3\u679c\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"*** Booting Zephyr OS build fd53c115d07a  ***\nthreadA: Hello World on csk6002_9s_nano!\nthreadB: Hello World on csk6002_9s_nano!\nthreadA: Hello World on csk6002_9s_nano!\nthreadA: Hello World on csk6002_9s_nano!\nthreadB: Hello World on csk6002_9s_nano!\nthreadA: Hello World on csk6002_9s_nano!\nthreadA: Hello World on csk6002_9s_nano!\nthreadB: Hello World on csk6002_9s_nano!\n")))}u.isMDXComponent=!0},93279:function(e,n,t){"use strict";n.Z=t.p+"assets/images/hello_world-70609b00276de9ad15c75adc08057a59.png"}}]);