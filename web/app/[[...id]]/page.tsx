import ChatApp from "@/components/ChatApp";

// One route for both "/" (new chat) and "/<chat id>". ChatApp reads the id from the URL.
export default function Page() {
  return <ChatApp />;
}
